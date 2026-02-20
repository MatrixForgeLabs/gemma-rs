//! SentencePiece tokenizer backed by kitoken.

const MOCK_TOKENIZER: &str = "unavailable";

pub struct Tokenizer {
    inner: kitoken::Kitoken,
}

impl Tokenizer {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes == MOCK_TOKENIZER.as_bytes() {
            return Err("tokenizer unavailable".to_string());
        }
        let inner = kitoken::Kitoken::from_sentencepiece_slice(bytes).map_err(|e| e.to_string())?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        match self.inner.encode(text, true) {
            Ok(tokens) => tokens.into_iter().map(|id| id as i32).collect(),
            Err(_) => Vec::new(),
        }
    }

    pub fn decode(&self, tokens: &[i32]) -> String {
        let ids: Vec<u32> = tokens.iter().map(|&id| id as u32).collect();
        match self.inner.decode(&ids, true) {
            Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
            Err(_) => String::new(),
        }
    }
}
