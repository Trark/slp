
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone)]
pub enum Extension {
    KhrFp16,
}

impl Extension {
    pub fn get_name(&self) -> &'static str {
        match *self {
            Extension::KhrFp16 => "cl_khr_fp16",
        }
    }
}
