use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CartridgeError {
    #[error("failed to read ROM: {0}")]
    Io(#[from] std::io::Error),
    #[error("ROM SHA-256 mismatch: got {got}, expected {expected}")]
    DigestMismatch { got: String, expected: &'static str },
    #[error("ROM is too small to contain an iNES header")]
    TooSmall,
    #[error("missing iNES magic bytes")]
    BadMagic,
    #[error("only mapper 0 / NROM is supported in the first fast path, got mapper {0}")]
    UnsupportedMapper(u8),
    #[error("invalid mapper 0 / NROM layout: PRG banks={prg_banks}, CHR banks={chr_banks}")]
    InvalidNromLayout { prg_banks: u8, chr_banks: u8 },
    #[error("ROM is truncated: expected at least {expected} bytes, got {actual}")]
    Truncated { expected: usize, actual: usize },
}

#[derive(Clone)]
pub struct Cartridge {
    pub prg_rom: Vec<u8>,
    pub chr_rom: Vec<u8>,
    pub vertical_mirroring: bool,
}

impl Cartridge {
    pub fn load_ines_for(
        path: impl AsRef<Path>,
        expected_digest: &'static str,
    ) -> Result<Self, CartridgeError> {
        let data = fs::read(path)?;
        assert_expected_rom_digest(&data, expected_digest)?;
        Self::parse_ines(&data)
    }

    pub fn parse_ines(data: &[u8]) -> Result<Self, CartridgeError> {
        if data.len() < 16 {
            return Err(CartridgeError::TooSmall);
        }
        if &data[0..4] != b"NES\x1a" {
            return Err(CartridgeError::BadMagic);
        }

        let prg_banks = data[4];
        let chr_banks = data[5];
        let flags6 = data[6];
        let flags7 = data[7];
        let mapper = (flags6 >> 4) | (flags7 & 0xf0);
        if mapper != 0 {
            return Err(CartridgeError::UnsupportedMapper(mapper));
        }
        if !matches!(prg_banks, 1 | 2) || !matches!(chr_banks, 0 | 1) {
            return Err(CartridgeError::InvalidNromLayout {
                prg_banks,
                chr_banks,
            });
        }

        let has_trainer = flags6 & 0b0000_0100 != 0;
        let mut offset = 16 + if has_trainer { 512 } else { 0 };
        let prg_len = usize::from(prg_banks) * 16 * 1024;
        let chr_len = usize::from(chr_banks) * 8 * 1024;
        let expected_len = offset + prg_len + chr_len;
        if data.len() < expected_len {
            return Err(CartridgeError::Truncated {
                expected: expected_len,
                actual: data.len(),
            });
        }

        let prg_rom = data[offset..offset + prg_len].to_vec();
        offset += prg_len;
        let chr_rom = if chr_len == 0 {
            vec![0; 8 * 1024]
        } else {
            data[offset..offset + chr_len].to_vec()
        };

        Ok(Self {
            prg_rom,
            chr_rom,
            vertical_mirroring: flags6 & 1 != 0,
        })
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn assert_expected_rom_digest(data: &[u8], expected: &'static str) -> Result<(), CartridgeError> {
    let digest = sha256_hex(data);
    if digest != expected {
        return Err(CartridgeError::DigestMismatch {
            got: digest,
            expected,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_ines(prg_banks: u8, chr_banks: u8) -> Vec<u8> {
        let mut data = b"NES\x1a".to_vec();
        data.extend([prg_banks, chr_banks, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        data.extend(vec![0; usize::from(prg_banks) * 16 * 1024]);
        data.extend(vec![0; usize::from(chr_banks) * 8 * 1024]);
        data
    }

    #[test]
    fn sha256_hex_matches_known_digest() {
        assert_eq!(
            sha256_hex(b"SuperMarioBros-Nes-turbo"),
            "f361123acc0af3ad64fae36fb1c7ce16ef0995cccaa5ef2cb7c8b3699518f112"
        );
    }

    #[test]
    fn load_ines_rejects_unexpected_rom_digest_before_header() {
        let path = std::env::temp_dir().join(format!(
            "supermariobrosnes-turbo-wrong-rom-{}.nes",
            std::process::id()
        ));
        fs::write(&path, b"not an iNES file").expect("write test ROM");
        const EXPECTED: &str = "not-the-real-digest";
        let err = match Cartridge::load_ines_for(&path, EXPECTED) {
            Ok(_) => panic!("wrong digest must be rejected"),
            Err(err) => err,
        };
        let _ = fs::remove_file(&path);
        assert!(matches!(err, CartridgeError::DigestMismatch { .. }));
        assert_eq!(
            err.to_string(),
            format!(
                "ROM SHA-256 mismatch: got {}, expected {EXPECTED}",
                sha256_hex(b"not an iNES file")
            )
        );
    }

    #[test]
    fn parser_rejects_invalid_and_truncated_nrom_layouts() {
        assert!(matches!(
            Cartridge::parse_ines(&synthetic_ines(3, 1)),
            Err(CartridgeError::InvalidNromLayout { .. })
        ));
        let mut truncated = synthetic_ines(1, 1);
        truncated.truncate(truncated.len() - 1);
        assert!(matches!(
            Cartridge::parse_ines(&truncated),
            Err(CartridgeError::Truncated { .. })
        ));

        let mut mapper_one = synthetic_ines(1, 1);
        mapper_one[6] = 0x10;
        assert!(matches!(
            Cartridge::parse_ines(&mapper_one),
            Err(CartridgeError::UnsupportedMapper(1))
        ));

        for (prg_banks, chr_banks) in [(1, 0), (1, 1), (2, 0), (2, 1)] {
            let cartridge = Cartridge::parse_ines(&synthetic_ines(prg_banks, chr_banks))
                .expect("valid NROM layout");
            assert_eq!(cartridge.prg_rom.len(), usize::from(prg_banks) * 16 * 1024);
            assert_eq!(cartridge.chr_rom.len(), 8 * 1024);
        }
    }
}
