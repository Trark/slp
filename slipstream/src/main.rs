
extern crate rustc_serialize;
extern crate docopt;
extern crate slp;

use std::io::Read;
use std::io::Write;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use docopt::Docopt;
use slp::FileLoader;

const USAGE: &'static str = "
Slipstream HLSL to OpenCL Compiler

Usage:
  slipstream [options] [-I <include_path>...] <source-file>
  slipstream --help

Options:
  -h --help                    Show help.
  --entry-point <entry_point>  Entry function [default: CSMAIN].
  -o <output_file>             Output file.
  -I <include_path>         Path to search for includes in.
";

#[derive(Debug, RustcDecodable)]
#[allow(non_snake_case)]
struct Args {
    flag_entry_point: String,
    flag_o: Option<String>,
    flag_I: Vec<String>,
    arg_source_file: String,
}

struct IncludeHandler {
    include_paths: Vec<PathBuf>,
}

impl FileLoader for IncludeHandler {
    fn load(&self, file_name: &str) -> Result<String, ()> {
        let file_path = Path::new(file_name);
        let mut file_res = File::open(file_path);
        if file_path.is_relative() {
            for path in &self.include_paths {
                match file_res {
                    Ok(_) => { },
                    Err(_) => {
                        let mut p = path.clone();
                        p.push(file_path);
                        file_res = File::open(p);
                    }
                }
            }
        }
        match file_res {
            Ok(mut file) => {
                let mut buffer = String::new();
                match file.read_to_string(&mut buffer) {
                    Ok(_) => Ok(buffer),
                    Err(_) => Err(()),
                }
            }
            Err(_) => Err(()),
        }
    }
}

fn main() {
    use slp::Input;
    use slp::hlsl_to_cl;

    let args: Args = Docopt::new(USAGE).and_then(|d| d.decode()).unwrap_or_else(|e| e.exit());
    let Args {
        flag_entry_point,
        flag_o: flag_output_file,
        flag_I: flag_include_paths,
        arg_source_file,
    } = args;

    let mut source_file = match File::open(&arg_source_file) {
        Ok(file) => file,
        Err(_) => {
            println!("Failed to load file '{}'", &arg_source_file);
            return;
        },
    };

    let mut source_contents = String::new();
    match source_file.read_to_string(&mut source_contents) {
        Ok(_) => { },
        Err(_) => {
            println!("Failed to read file '{}'", &arg_source_file);
            return;
        },
    };

    let paths = flag_include_paths.iter().map(|s| PathBuf::from(s)).collect::<Vec<_>>();
    for path in &paths {
        if !path.is_dir() {
            let s = match path.to_str() {
                Some(s) => s,
                None => "<invalid>",
            };
            println!("Include path '{}' is not a directory", s);
            return;
        }
    }

    let include_handler = Box::new(IncludeHandler { include_paths: paths });

    let input = Input {
        entry_point: flag_entry_point,
        main_file: source_contents,
        file_loader: include_handler,
    };

    let output_res = hlsl_to_cl(input);
    match output_res {
        Ok(output) => {
            match flag_output_file {
                Some(output_file) => {
                    let mut file = match File::create(&output_file) {
                        Ok(file) => file,
                        Err(_) => {
                            println!("Failed to open output file {}", output_file);
                            return;
                        }
                    };
                    match file.write(output.code.to_string().as_bytes()) {
                        Ok(_) => { },
                        Err(_) => println!("Failed to write to output file"),
                    };
                }
                None => {
                    println!("{}", output.code.to_string())
                }
            }
        }
        Err(err) => println!("Error: {:?}", err),
    };

}
