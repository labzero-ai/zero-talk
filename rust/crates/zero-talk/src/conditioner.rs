//! The conditioner: from the acoustic style, the token histogram and the statistics of the student's latents over the whole
//! sentence, it predicts the generator's 48 static norm sites' means and variances, so the generator needs no activation statistics
//! at run time. Features (1,330) are standardized, reduced to rank 8 and expanded to 9,216 channels, with one head for the means and
//! one for the scales.
use std::collections::HashMap;

use crate::archive::{Part, Result};
use crate::kernels::Act;
use crate::plan::{fold_combine, COORDS};

/// The statistics block: for each channel of `latents` the mean over its frames, then `ln(population sd + 1e-4)`; all means come first.
/// The sums run in f64, so every platform gives the same numbers.
pub fn latent_statistics(latents: &Act) -> Vec<f32> {
    let n = latents.t as f64;
    let (mut mean, mut log_sd) = (Vec::with_capacity(latents.c), Vec::with_capacity(latents.c));
    for c in 0..latents.c {
        let row = latents.row(c);
        let m = row.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = row.iter().map(|&v| (v as f64 - m) * (v as f64 - m)).sum::<f64>() / n;
        mean.push(m as f32);
        log_sd.push((var.sqrt() + 1e-4).ln() as f32);
    }
    mean.extend(log_sd);
    mean
}

pub struct Conditioner {
    center: Vec<f32>,
    std: Vec<f32>,
    a_mean: Vec<f32>, // [features][rank]
    a_scale: Vec<f32>,
    b_mean: Vec<f32>, // [rank][channels]
    b_scale: Vec<f32>,
    center_mean: Vec<f32>,
    center_scale: Vec<f32>,
    epsilon: Vec<f32>,
    voice_mean: Vec<f32>, // [1 + COORDS][rank]: the voice features' share of each head's bottleneck
    voice_scale: Vec<f32>,
    vocab: usize,
    rank: usize,
    sites: Vec<(String, usize, usize)>,
    index: HashMap<String, usize>,
}

/// Predicted moments of every site, concatenated in the conditioner's site order.
pub struct Moments {
    pub mean: Vec<f32>,
    pub variance: Vec<f32>,
}

impl Conditioner {
    pub fn load(ar: &Part) -> Result<Self> {
        let sites = ar.meta("conditioner_sites")?.split(';').map(|s| {
            let mut f = s.rsplitn(3, ':');
            let (b, a, n) = (f.next().ok_or("site")?, f.next().ok_or("site")?, f.next().ok_or("site")?);
            Ok((n.to_string(), a.parse().map_err(|_| "site start")?, b.parse().map_err(|_| "site end")?))
        }).collect::<Result<Vec<(String, usize, usize)>>>()?;
        let channels: usize = ar.meta_as("conditioner_channels")?;
        match ar.meta("conditioner_feature_kind") {
            Ok("latent-stats") => {}
            Err(_) | Ok("text") => return Err("unsupported model file: this build reads Zero Talk 11M model files".into()),
            Ok(k) => return Err(format!("unknown conditioner feature kind '{k}'")),
        }
        let (center, rank, vocab): (_, _, usize) = (ar.f32("feature_center")?, ar.shape("projection_B_mean")?[0], ar.meta_as("conditioner_vocab_size")?);
        // The voice features are folded into the length coordinates, so the vector is the histogram and the statistics.
        if sites.last().map(|s| s.2) != Some(channels) || ar.shape("projection_A_mean")? != [center.len(), rank]
            || center.len() != vocab + ar.meta_as::<usize>("conditioner_text_width")? {
            return Err("inconsistent conditioner".into());
        }
        Ok(Conditioner {
            std: ar.f32("feature_std")?,
            a_mean: ar.f32("projection_A_mean")?,
            a_scale: ar.f32("projection_A_scale")?,
            b_mean: ar.f32("projection_B_mean")?,
            b_scale: ar.f32("projection_B_scale")?,
            center_mean: ar.f32("target_center_mean")?,
            center_scale: ar.f32("target_center_scale")?,
            epsilon: ar.f32("epsilon")?,
            voice_mean: ar.f32("voice_fold_mean")?,
            voice_scale: ar.f32("voice_fold_scale")?,
            vocab,
            center,
            rank,
            index: sites.iter().enumerate().map(|(i, s)| (s.0.clone(), i)).collect(),
            sites,
        })
    }

    /// `coords`: the sentence's length coordinates (the style features are folded into them); `statistics`: the student's latents'
    /// `latent_statistics` over the whole chunk.
    pub fn moments(&self, coords: &[f32; COORDS], ids: &[usize], statistics: &[f32]) -> Result<Moments> {
        let mut feat = Vec::new();
        let mut hist = vec![0f64; self.vocab];
        for &id in ids {
            *hist.get_mut(id).ok_or("token id out of range")? += 1.0;
        }
        feat.extend(hist.iter().map(|h| (h / ids.len() as f64) as f32));
        feat.extend_from_slice(statistics);
        if feat.len() != self.center.len() {
            return Err(format!("expected {} features, got {}", self.center.len(), feat.len()));
        }
        let x: Vec<f32> = feat.iter().zip(&self.center).zip(&self.std).map(|((f, c), s)| (f - c) / s).collect();
        let channels = self.epsilon.len();
        let project = |a: &[f32], style: &[f32], b: &[f32], center: &[f32]| -> Vec<f32> {
            let mut z: Vec<f32> = fold_combine(style, coords).into_iter().map(|v| v as f32).collect();
            for (i, xi) in x.iter().enumerate() {
                for (zr, ar) in z.iter_mut().zip(&a[i * self.rank..]) {
                    *zr += xi * ar;
                }
            }
            let mut out = center.to_vec();
            for (r, zr) in z.iter().enumerate() {
                for (o, br) in out.iter_mut().zip(&b[r * channels..(r + 1) * channels]) {
                    *o += zr * br;
                }
            }
            out
        };
        let mean = project(&self.a_mean, &self.voice_mean, &self.b_mean, &self.center_mean);
        let log_den = project(&self.a_scale, &self.voice_scale, &self.b_scale, &self.center_scale);
        let variance = log_den.iter().zip(&self.epsilon).map(|(l, e)| ((2.0 * l).exp() - e).max(0.0)).collect();
        Ok(Moments { mean, variance })
    }

    /// The (mean, variance) of the named norm sites (e.g. `resblocks.0.adain1.0.norm`), in the order given.
    pub fn sites<'a>(&self, m: &Moments, names: impl Iterator<Item = &'a str>) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        names.map(|n| {
            let &i = self.index.get(n).ok_or(format!("no conditioner site '{n}'"))?;
            let (_, a, b) = self.sites[i];
            Ok((m.mean[a..b].to_vec(), m.variance[a..b].to_vec()))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistics_are_means_then_log_population_sds() {
        // two channels over four frames: 1,2,3,4 (mean 2.5, population variance 1.25) and a constant 7 (sd 0, only the 1e-4 is left)
        let latents = Act { c: 2, t: 4, data: vec![1.0, 2.0, 3.0, 4.0, 7.0, 7.0, 7.0, 7.0] };
        let s = latent_statistics(&latents);
        assert_eq!(s.len(), 4);
        assert_eq!((s[0], s[1]), (2.5, 7.0));
        assert!((s[2] as f64 - (1.25f64.sqrt() + 1e-4).ln()).abs() < 1e-6);
        assert!((s[3] as f64 - 1e-4f64.ln()).abs() < 1e-6);
    }
}
