//! Simple command-line args helper.

use crate::basics::Tristate;
use gemma_io::io::Path;

pub trait Args: Sized {
    fn for_each<V: ArgVisitor>(&mut self, visitor: &mut V);

    fn init(&mut self) {
        let mut visitor = InitVisitor;
        self.for_each(&mut visitor);
    }

    fn help(&mut self) {
        let mut visitor = HelpVisitor;
        self.for_each(&mut visitor);
    }

    fn print(&mut self, verbosity: i32) {
        let mut visitor = PrintVisitor { verbosity };
        self.for_each(&mut visitor);
    }

    fn parse(&mut self, args: &[String]) {
        let mut visitor = ParseVisitor { args };
        self.for_each(&mut visitor);
    }

    fn init_and_parse(&mut self, args: &[String]) {
        self.init();
        self.parse(args);
    }
}

pub trait ArgVisitor {
    fn visit_u64(
        &mut self,
        value: &mut u64,
        name: &str,
        init: u64,
        help: &str,
        print_verbosity: i32,
    );
    fn visit_usize(
        &mut self,
        value: &mut usize,
        name: &str,
        init: usize,
        help: &str,
        print_verbosity: i32,
    );
    fn visit_i32(
        &mut self,
        value: &mut i32,
        name: &str,
        init: i32,
        help: &str,
        print_verbosity: i32,
    );
    fn visit_f32(
        &mut self,
        value: &mut f32,
        name: &str,
        init: f32,
        help: &str,
        print_verbosity: i32,
    );
    fn visit_bool(
        &mut self,
        value: &mut bool,
        name: &str,
        init: bool,
        help: &str,
        print_verbosity: i32,
    );
    fn visit_string(
        &mut self,
        value: &mut String,
        name: &str,
        init: &str,
        help: &str,
        print_verbosity: i32,
    );
    fn visit_path(
        &mut self,
        value: &mut Path,
        name: &str,
        init: &str,
        help: &str,
        print_verbosity: i32,
    );
    fn visit_tristate(
        &mut self,
        value: &mut Tristate,
        name: &str,
        init: Tristate,
        help: &str,
        print_verbosity: i32,
    );
}

struct InitVisitor;

impl ArgVisitor for InitVisitor {
    fn visit_u64(&mut self, value: &mut u64, _name: &str, init: u64, _help: &str, _p: i32) {
        *value = init;
    }
    fn visit_usize(&mut self, value: &mut usize, _name: &str, init: usize, _help: &str, _p: i32) {
        *value = init;
    }
    fn visit_i32(&mut self, value: &mut i32, _name: &str, init: i32, _help: &str, _p: i32) {
        *value = init;
    }
    fn visit_f32(&mut self, value: &mut f32, _name: &str, init: f32, _help: &str, _p: i32) {
        *value = init;
    }
    fn visit_bool(&mut self, value: &mut bool, _name: &str, init: bool, _help: &str, _p: i32) {
        *value = init;
    }
    fn visit_string(&mut self, value: &mut String, _name: &str, init: &str, _help: &str, _p: i32) {
        *value = init.to_string();
    }
    fn visit_path(&mut self, value: &mut Path, _name: &str, init: &str, _help: &str, _p: i32) {
        value.path = init.to_string();
    }
    fn visit_tristate(
        &mut self,
        value: &mut Tristate,
        _name: &str,
        init: Tristate,
        _help: &str,
        _p: i32,
    ) {
        *value = init;
    }
}

struct HelpVisitor;

impl ArgVisitor for HelpVisitor {
    fn visit_u64(&mut self, _value: &mut u64, name: &str, _init: u64, help: &str, _p: i32) {
        eprintln!("  --{} : {}", name, help);
    }
    fn visit_usize(&mut self, _value: &mut usize, name: &str, _init: usize, help: &str, _p: i32) {
        eprintln!("  --{} : {}", name, help);
    }
    fn visit_i32(&mut self, _value: &mut i32, name: &str, _init: i32, help: &str, _p: i32) {
        eprintln!("  --{} : {}", name, help);
    }
    fn visit_f32(&mut self, _value: &mut f32, name: &str, _init: f32, help: &str, _p: i32) {
        eprintln!("  --{} : {}", name, help);
    }
    fn visit_bool(&mut self, _value: &mut bool, name: &str, _init: bool, help: &str, _p: i32) {
        eprintln!("  --{} : {}", name, help);
    }
    fn visit_string(&mut self, _value: &mut String, name: &str, _init: &str, help: &str, _p: i32) {
        eprintln!("  --{} : {}", name, help);
    }
    fn visit_path(&mut self, _value: &mut Path, name: &str, _init: &str, help: &str, _p: i32) {
        eprintln!("  --{} : {}", name, help);
    }
    fn visit_tristate(
        &mut self,
        _value: &mut Tristate,
        name: &str,
        _init: Tristate,
        help: &str,
        _p: i32,
    ) {
        eprintln!("  --{} : {}", name, help);
    }
}

struct PrintVisitor {
    verbosity: i32,
}

impl ArgVisitor for PrintVisitor {
    fn visit_u64(
        &mut self,
        value: &mut u64,
        name: &str,
        _init: u64,
        _help: &str,
        print_verbosity: i32,
    ) {
        if self.verbosity >= print_verbosity {
            eprintln!("{:<30}: {}", name, value);
        }
    }
    fn visit_usize(
        &mut self,
        value: &mut usize,
        name: &str,
        _init: usize,
        _help: &str,
        print_verbosity: i32,
    ) {
        if self.verbosity >= print_verbosity {
            eprintln!("{:<30}: {}", name, value);
        }
    }
    fn visit_i32(
        &mut self,
        value: &mut i32,
        name: &str,
        _init: i32,
        _help: &str,
        print_verbosity: i32,
    ) {
        if self.verbosity >= print_verbosity {
            eprintln!("{:<30}: {}", name, value);
        }
    }
    fn visit_f32(
        &mut self,
        value: &mut f32,
        name: &str,
        _init: f32,
        _help: &str,
        print_verbosity: i32,
    ) {
        if self.verbosity >= print_verbosity {
            eprintln!("{:<30}: {}", name, value);
        }
    }
    fn visit_bool(
        &mut self,
        value: &mut bool,
        name: &str,
        _init: bool,
        _help: &str,
        print_verbosity: i32,
    ) {
        if self.verbosity >= print_verbosity {
            eprintln!("{:<30}: {}", name, value);
        }
    }
    fn visit_string(
        &mut self,
        value: &mut String,
        name: &str,
        _init: &str,
        _help: &str,
        print_verbosity: i32,
    ) {
        if self.verbosity >= print_verbosity {
            eprintln!("{:<30}: {}", name, value);
        }
    }
    fn visit_path(
        &mut self,
        value: &mut Path,
        name: &str,
        _init: &str,
        _help: &str,
        print_verbosity: i32,
    ) {
        if self.verbosity >= print_verbosity {
            eprintln!("{:<30}: {}", name, value.shortened());
        }
    }
    fn visit_tristate(
        &mut self,
        value: &mut Tristate,
        name: &str,
        _init: Tristate,
        _help: &str,
        print_verbosity: i32,
    ) {
        if self.verbosity >= print_verbosity {
            eprintln!("{:<30}: {}", name, value.as_str());
        }
    }
}

struct ParseVisitor<'a> {
    args: &'a [String],
}

impl<'a> ParseVisitor<'a> {
    fn find_value(&self, name: &str) -> Option<&str> {
        let prefixed = format!("--{}", name);
        let mut it = self.args.iter();
        while let Some(arg) = it.next() {
            if arg == &prefixed {
                return it.next().map(|s| s.as_str());
            }
        }
        None
    }
}

impl<'a> ArgVisitor for ParseVisitor<'a> {
    fn visit_u64(&mut self, value: &mut u64, name: &str, _init: u64, _help: &str, _p: i32) {
        if let Some(v) = self.find_value(name) {
            *value = v.parse().unwrap();
        }
    }
    fn visit_usize(&mut self, value: &mut usize, name: &str, _init: usize, _help: &str, _p: i32) {
        if let Some(v) = self.find_value(name) {
            *value = v.parse().unwrap();
        }
    }
    fn visit_i32(&mut self, value: &mut i32, name: &str, _init: i32, _help: &str, _p: i32) {
        if let Some(v) = self.find_value(name) {
            *value = v.parse().unwrap();
        }
    }
    fn visit_f32(&mut self, value: &mut f32, name: &str, _init: f32, _help: &str, _p: i32) {
        if let Some(v) = self.find_value(name) {
            *value = v.parse().unwrap();
        }
    }
    fn visit_bool(&mut self, value: &mut bool, name: &str, _init: bool, _help: &str, _p: i32) {
        if let Some(v) = self.find_value(name) {
            *value = matches!(v.to_ascii_lowercase().as_str(), "true" | "1" | "on");
        }
    }
    fn visit_string(&mut self, value: &mut String, name: &str, _init: &str, _help: &str, _p: i32) {
        if let Some(v) = self.find_value(name) {
            *value = v.to_string();
        }
    }
    fn visit_path(&mut self, value: &mut Path, name: &str, _init: &str, _help: &str, _p: i32) {
        if let Some(v) = self.find_value(name) {
            value.path = v.to_string();
        }
    }
    fn visit_tristate(
        &mut self,
        value: &mut Tristate,
        name: &str,
        _init: Tristate,
        _help: &str,
        _p: i32,
    ) {
        if let Some(v) = self.find_value(name) {
            let val = v.to_ascii_lowercase();
            *value = match val.as_str() {
                "true" | "1" | "on" => Tristate::True,
                "false" | "0" | "off" => Tristate::False,
                "default" | "auto" | "-1" => Tristate::Default,
                _ => *value,
            };
        }
    }
}

pub fn has_help(args: &[String]) -> bool {
    if args.len() <= 1 {
        return true;
    }
    args.iter().any(|a| a == "--help")
}

pub fn abort_if_invalid_args<T: Args>(args: &mut T, err: Option<&str>) {
    if let Some(msg) = err {
        args.help();
        panic!("Problem with args: {}", msg);
    }
}
