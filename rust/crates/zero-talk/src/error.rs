//! The error type of the library.
use std::fmt;

/// An error from Zero Talk. The message says what went wrong and, where it can, what to change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    message: String,
}

/// The result type of the library: `Result<T, zero_talk::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Error { message: message.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

impl From<String> for Error {
    fn from(message: String) -> Self {
        Error { message }
    }
}
