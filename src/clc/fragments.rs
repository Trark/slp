
use super::cir::*;

/// OpenCL function fragments
/// For helper functions required by the transpiler. Used Fragments will be
/// detected and emitted at the top of the module.
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Clone)]
pub enum Fragment {
    VectorCast(Scalar, Scalar, VectorDimension),
}

impl Fragment {
    pub fn generate(&self, name: &str) -> FunctionDefinition {
        match *self {
            Fragment::VectorCast(ref from, ref to, ref dim) => {
                let from_param = "from".to_string();
                let to_local = "to".to_string();
                let mut body = vec![Statement::Var(VarDef { name: to_local.clone(), typename: Type::Vector(to.clone(), dim.clone()), assignment: None } )];
                for i in 0..(dim.as_u32()) {
                    let index = i as u64;
                    body.push(Statement::Expression(Expression::BinaryOperation(BinOp::Assignment,
                        Box::new(Expression::ArraySubscript(Box::new(Expression::Variable(to_local.clone())), Box::new(Expression::Literal(Literal::Int(index))))),
                        Box::new(Expression::ArraySubscript(Box::new(Expression::Variable(from_param.clone())), Box::new(Expression::Literal(Literal::Int(index))))),
                    )));
                }
                body.push(Statement::Return(Expression::Variable(to_local.clone())));
                FunctionDefinition {
                    name: name.to_string(),
                    returntype: Type::Vector(to.clone(), dim.clone()),
                    params: vec![FunctionParam { name: from_param, typename: Type::Vector(from.clone(), dim.clone()) }],
                    body: body,
                }
            }
        }
    }

    fn get_type_candidate_name(ty: &Scalar) -> &'static str {
        match *ty {
            Scalar::Char => "char",
            Scalar::UChar => "uchar",
            Scalar::Short => "short",
            Scalar::UShort => "ushort",
            Scalar::Int => "int",
            Scalar::UInt => "uint",
            Scalar::Long => "long",
            Scalar::ULong => "ulong",
            Scalar::Half => "half",
            Scalar::Float => "float",
            Scalar::Double => "double",
        }
    }

    fn get_dimension_candidate_name(dim: &VectorDimension) -> &'static str {
        match *dim {
            VectorDimension::Two => "2",
            VectorDimension::Three => "3",
            VectorDimension::Four => "4",
            VectorDimension::Eight => "8",
            VectorDimension::Sixteen => "16",
        }
    }

    pub fn get_candidate_name(&self) -> String {
        match *self {
            Fragment::VectorCast(ref from, ref to, ref dim) => {
                let from_str = Fragment::get_type_candidate_name(from);
                let to_str = Fragment::get_type_candidate_name(to);
                let dim_str = Fragment::get_dimension_candidate_name(dim);
                format!("cast_{}_to_{}_{}", from_str, to_str, dim_str)
            }
        }
    }
}