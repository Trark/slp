
use pel;
use typer::TyperError;
use typer::FunctionName;
use typer::FunctionOverload;
use typer::UnresolvedFunction;
use typer::ErrorType;
use typer::ToErrorType;
use typer::parse_datalayout;
use typer::parse_type;
use typer::ExpressionContext;
use typer::VariableExpression;
use casting::ImplicitConversion;
use casting::ConversionPriority;
use intrinsics;
use intrinsics::IntrinsicFactory;
use slp_lang_hst as hst;
use slp_lang_hir as hir;
use slp_lang_hir::ExpressionType;
use slp_lang_hir::ToExpressionType;
use slp_lang_hir::Intrinsic;

pub type ClassType = hir::Type;

#[derive(PartialEq, Debug, Clone)]
pub struct UnresolvedMethod(pub String,
                            pub ClassType,
                            pub Vec<FunctionOverload>,
                            pub pel::Expression);

#[derive(PartialEq, Debug, Clone)]
pub enum TypedExpression {
    // Expression + Type
    Value(pel::Expression, ExpressionType),
    // Name of function + overloads
    Function(UnresolvedFunction),
    // Name of function + overloads + object
    Method(UnresolvedMethod),
}

impl ToErrorType for TypedExpression {
    fn to_error_type(&self) -> ErrorType {
        match *self {
            TypedExpression::Value(_, ref ety) => ety.to_error_type(),
            TypedExpression::Function(UnresolvedFunction(ref name, ref overloads)) => {
                ErrorType::Function(name.clone(), overloads.clone())
            }
            TypedExpression::Method(UnresolvedMethod(ref name, ref ct, ref overloads, _)) => {
                ErrorType::Method(name.clone(), ct.clone(), overloads.clone())
            }
        }
    }
}

fn parse_variable(name: &String,
                  context: &ExpressionContext)
                  -> Result<TypedExpression, TyperError> {
    Ok(match try!(context.find_variable(name)) {
        VariableExpression::Local(var, ty) => {
            TypedExpression::Value(pel::Expression::Variable(var), ty.to_lvalue())
        }
        VariableExpression::Global(id, ty) => {
            TypedExpression::Value(pel::Expression::Global(id), ty.to_lvalue())
        }
        VariableExpression::Constant(id, name, ty) => {
            TypedExpression::Value(pel::Expression::ConstantVariable(id, name), ty.to_lvalue())
        }
        VariableExpression::Function(func) => TypedExpression::Function(func),
    })
}

fn find_function_type(overloads: &Vec<FunctionOverload>,
                      param_types: &[ExpressionType])
                      -> Result<(FunctionOverload, Vec<ImplicitConversion>), TyperError> {
    use casting::VectorRank;
    fn find_overload_casts(overload: &FunctionOverload,
                           param_types: &[ExpressionType])
                           -> Result<Vec<ImplicitConversion>, ()> {
        let mut overload_casts = Vec::with_capacity(param_types.len());
        for (required_type, source_type) in overload.2.iter().zip(param_types.iter()) {
            let &hir::ParamType(ref ty, ref it, ref interp) = required_type;

            let ety = match *it {
                hir::InputModifier::In => ty.to_rvalue(),
                hir::InputModifier::Out |
                hir::InputModifier::InOut => ty.to_lvalue(),
            };
            match *interp {
                Some(_) => return Err(()),
                None => {}
            };

            if let Ok(cast) = ImplicitConversion::find(source_type, &ety) {
                overload_casts.push(cast)
            } else {
                return Err(());
            }
        }
        Ok(overload_casts)
    }

    let mut casts = Vec::with_capacity(overloads.len());
    for overload in overloads {
        if param_types.len() == overload.2.len() {
            if let Ok(param_casts) = find_overload_casts(overload, param_types) {
                casts.push((overload.clone(), param_casts))
            }
        }
    }

    // Cull everything that isn't equal best at matching numeric type
    let mut winning_numeric_casts = Vec::with_capacity(1);
    for &(ref candidate, ref candidate_casts) in &casts {
        let mut winning = true;
        for &(ref against, ref against_casts) in &casts {
            if candidate == against {
                continue;
            }
            assert_eq!(candidate_casts.len(), against_casts.len());
            let mut not_worse_than = true;
            for (candidate_cast, against_cast) in candidate_casts.iter().zip(against_casts) {
                let candidate_rank = candidate_cast.get_rank().get_numeric_rank().clone();
                let against_rank = against_cast.get_rank().get_numeric_rank().clone();
                match candidate_rank.compare(&against_rank) {
                    ConversionPriority::Better => {}
                    ConversionPriority::Equal => {}
                    ConversionPriority::Worse => not_worse_than = false,
                };
            }
            if !not_worse_than {
                winning = false;
                break;
            }
        }
        if winning {
            winning_numeric_casts.push((candidate.clone(), candidate_casts.clone()));
        }
    }

    if winning_numeric_casts.len() > 0 {

        fn count_by_rank(casts: &[ImplicitConversion], rank: &VectorRank) -> usize {
            casts.iter()
                .filter(|ref cast| cast.get_rank().get_vector_rank() == rank)
                .count()
        }

        let map_order = |(overload, casts): (_, Vec<ImplicitConversion>)| {
            let order = VectorRank::worst_to_best()
                .iter()
                .map(|rank| count_by_rank(&casts, rank))
                .collect::<Vec<_>>();
            (overload, casts, order)
        };

        let casts = winning_numeric_casts.into_iter().map(map_order).collect::<Vec<_>>();;

        let mut best_order = casts[0].2.clone();
        for &(_, _, ref order) in &casts {
            if *order < best_order {
                best_order = order.clone();
            }
        }

        let casts = casts.into_iter()
            .filter(|&(_, _, ref order)| *order == best_order)
            .collect::<Vec<_>>();

        if casts.len() == 1 {
            let (candidate, casts, _) = casts[0].clone();
            return Ok((candidate, casts));
        }
    }

    Err(TyperError::FunctionArgumentTypeMismatch(overloads.clone(), param_types.to_vec()))
}

fn apply_casts(casts: Vec<ImplicitConversion>,
               values: Vec<pel::Expression>)
               -> Vec<pel::Expression> {
    assert_eq!(casts.len(), values.len());
    values.into_iter()
        .enumerate()
        .map(|(index, value)| casts[index].apply_pel(value))
        .collect::<Vec<_>>()
}

fn write_function(unresolved: UnresolvedFunction,
                  param_types: &[ExpressionType],
                  param_values: Vec<pel::Expression>)
                  -> Result<TypedExpression, TyperError> {
    // Find the matching function overload
    let (FunctionOverload(name, return_type_ty, _), casts) =
        try!(find_function_type(&unresolved.1, param_types));
    // Apply implicit casts
    let param_values = apply_casts(casts, param_values);
    let return_type = return_type_ty.to_rvalue();

    match name {
        FunctionName::Intrinsic(factory) => {
            Ok(TypedExpression::Value(factory.create_intrinsic(&param_values), return_type))
        }
        FunctionName::User(id) => {
            Ok(TypedExpression::Value(pel::Expression::Call(id, param_values), return_type))
        }
    }
}

fn write_method(unresolved: UnresolvedMethod,
                param_types: &[ExpressionType],
                param_values: Vec<pel::Expression>)
                -> Result<TypedExpression, TyperError> {
    // Find the matching method overload
    let (FunctionOverload(name, return_type_ty, _), casts) =
        try!(find_function_type(&unresolved.2, param_types));
    // Apply implicit casts
    let mut param_values = apply_casts(casts, param_values);
    // Add struct as implied first argument
    param_values.insert(0, unresolved.3);
    let return_type = return_type_ty.to_rvalue();

    match name {
        FunctionName::Intrinsic(factory) => {
            Ok(TypedExpression::Value(factory.create_intrinsic(&param_values), return_type))
        }
        FunctionName::User(_) => panic!("User defined methods should not exist"),
    }
}

fn parse_literal(ast: &hst::Literal) -> Result<TypedExpression, TyperError> {
    match ast {
        &hst::Literal::Bool(b) => {
            Ok(TypedExpression::Value(pel::Expression::Literal(hir::Literal::Bool(b)),
                                      hir::Type::bool().to_rvalue()))
        }
        &hst::Literal::UntypedInt(i) => {
            Ok(TypedExpression::Value(pel::Expression::Literal(hir::Literal::UntypedInt(i)),
                                      hir::Type::from_scalar(hir::ScalarType::UntypedInt)
                                          .to_rvalue()))
        }
        &hst::Literal::Int(i) => {
            Ok(TypedExpression::Value(pel::Expression::Literal(hir::Literal::Int(i)),
                                      hir::Type::int().to_rvalue()))
        }
        &hst::Literal::UInt(i) => {
            Ok(TypedExpression::Value(pel::Expression::Literal(hir::Literal::UInt(i)),
                                      hir::Type::uint().to_rvalue()))
        }
        &hst::Literal::Long(i) => {
            Ok(TypedExpression::Value(pel::Expression::Literal(hir::Literal::Long(i)),
                                      hir::Type::from_scalar(hir::ScalarType::UntypedInt)
                                          .to_rvalue()))
        }
        &hst::Literal::Half(f) => {
            Ok(TypedExpression::Value(pel::Expression::Literal(hir::Literal::Half(f)),
                                      hir::Type::float().to_rvalue()))
        }
        &hst::Literal::Float(f) => {
            Ok(TypedExpression::Value(pel::Expression::Literal(hir::Literal::Float(f)),
                                      hir::Type::float().to_rvalue()))
        }
        &hst::Literal::Double(f) => {
            Ok(TypedExpression::Value(pel::Expression::Literal(hir::Literal::Double(f)),
                                      hir::Type::double().to_rvalue()))
        }
    }
}

fn parse_expr_unaryop(op: &hst::UnaryOp,
                      expr: &hst::Expression,
                      context: &ExpressionContext)
                      -> Result<TypedExpression, TyperError> {
    match try!(parse_expr(expr, context)) {
        TypedExpression::Value(expr_ir, expr_ty) => {
            fn enforce_increment_type(ety: &ExpressionType,
                                      op: &hst::UnaryOp)
                                      -> Result<(), TyperError> {
                match *ety {
                    ExpressionType(_, hir::ValueType::Rvalue) => {
                        Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown))
                    }
                    ExpressionType(hir::Type(hir::TypeLayout::Scalar(hir::ScalarType::Bool),
                                             _),
                                   _) => {
                        Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown))
                    }
                    ExpressionType(hir::Type(hir::TypeLayout::Vector(hir::ScalarType::Bool,
                                                                     _),
                                             _),
                                   _) => {
                        Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown))
                    }
                    ExpressionType(hir::Type(hir::TypeLayout::Matrix(hir::ScalarType::Bool,
                                                                     _,
                                                                     _),
                                             _),
                                   _) => {
                        Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown))
                    }
                    _ => Ok(()),
                }
            }
            let (intrinsic, eir, ety) = match *op {
                hst::UnaryOp::PrefixIncrement => {
                    try!(enforce_increment_type(&expr_ty, op));
                    (hir::Intrinsic1::PrefixIncrement(expr_ty.0.clone()), expr_ir, expr_ty)
                }
                hst::UnaryOp::PrefixDecrement => {
                    try!(enforce_increment_type(&expr_ty, op));
                    (hir::Intrinsic1::PrefixDecrement(expr_ty.0.clone()), expr_ir, expr_ty)
                }
                hst::UnaryOp::PostfixIncrement => {
                    try!(enforce_increment_type(&expr_ty, op));
                    (hir::Intrinsic1::PostfixIncrement(expr_ty.0.clone()), expr_ir, expr_ty)
                }
                hst::UnaryOp::PostfixDecrement => {
                    try!(enforce_increment_type(&expr_ty, op));
                    (hir::Intrinsic1::PostfixDecrement(expr_ty.0.clone()), expr_ir, expr_ty)
                }
                hst::UnaryOp::Plus => {
                    (hir::Intrinsic1::Plus(expr_ty.0.clone()), expr_ir, expr_ty.0.to_rvalue())
                }
                hst::UnaryOp::Minus => {
                    (hir::Intrinsic1::Minus(expr_ty.0.clone()), expr_ir, expr_ty.0.to_rvalue())
                }
                hst::UnaryOp::LogicalNot => {
                    let ty = match expr_ty.0 {
                        hir::Type(hir::TypeLayout::Scalar(_), _) => {
                            hir::Type::from_layout(hir::TypeLayout::Scalar(hir::ScalarType::Bool))
                        }
                        hir::Type(hir::TypeLayout::Vector(_, x), _) => {
                            hir::Type::from_layout(hir::TypeLayout::Vector(hir::ScalarType::Bool,
                                                                           x))
                        }
                        hir::Type(hir::TypeLayout::Matrix(_, x, y), _) => {
                            hir::Type::from_layout(hir::TypeLayout::Matrix(hir::ScalarType::Bool,
                                                                           x,
                                                                           y))
                        }
                        _ => {
                            return Err(TyperError::UnaryOperationWrongTypes(op.clone(),
                                                                            ErrorType::Unknown))
                        }
                    };
                    let ety = ty.clone().to_rvalue();
                    (hir::Intrinsic1::LogicalNot(ty), expr_ir, ety)
                }
                hst::UnaryOp::BitwiseNot => {
                    match (expr_ty.0).0 {
                        hir::TypeLayout::Scalar(hir::ScalarType::Int) |
                        hir::TypeLayout::Scalar(hir::ScalarType::UInt) => {
                            (hir::Intrinsic1::BitwiseNot(expr_ty.0.clone()),
                             expr_ir,
                             expr_ty.0.to_rvalue())
                        }
                        _ => {
                            return Err(TyperError::UnaryOperationWrongTypes(op.clone(),
                                                                            ErrorType::Unknown))
                        }
                    }
                }
            };
            Ok(TypedExpression::Value(pel::Expression::Intrinsic1(intrinsic, Box::new(eir)), ety))
        }
        _ => Err(TyperError::UnaryOperationWrongTypes(op.clone(), ErrorType::Unknown)),
    }
}

fn most_sig_type_dim(lhs: &hir::TypeLayout,
                     rhs: &hir::TypeLayout)
                     -> Option<hir::NumericDimension> {
    use slp_lang_hir::TypeLayout::*;
    use std::cmp::min;
    use std::cmp::max;
    match (lhs, rhs) {
        (&Scalar(_), &Scalar(_)) => Some(hir::NumericDimension::Scalar),
        (&Scalar(_), &Vector(_, ref x)) => Some(hir::NumericDimension::Vector(*x)),
        (&Vector(_, ref x), &Scalar(_)) => Some(hir::NumericDimension::Vector(*x)),
        (&Vector(_, ref x1), &Vector(_, ref x2)) if *x1 == 1 || *x2 == 1 => {
            Some(hir::NumericDimension::Vector(max(*x1, *x2)))
        }
        (&Vector(_, ref x1), &Vector(_, ref x2)) => {
            let x = min(*x1, *x2);
            Some(hir::NumericDimension::Vector(x))
        }
        (&Matrix(_, ref x1, ref y1), &Matrix(_, ref x2, ref y2)) => {
            let x = min(*x1, *x2);
            let y = min(*y1, *y2);
            Some(hir::NumericDimension::Matrix(x, y))
        }
        _ => None,
    }
}

fn most_sig_scalar(left: &hir::ScalarType, right: &hir::ScalarType) -> hir::ScalarType {
    use slp_lang_hir::ScalarType;

    // The limited number of hlsl types means these happen to always have one
    // type being the common type
    fn get_order(ty: &ScalarType) -> Option<u32> {
        match *ty {
            ScalarType::Bool => Some(0),
            ScalarType::Int => Some(1),
            ScalarType::UInt => Some(2),
            ScalarType::Half => Some(3),
            ScalarType::Float => Some(4),
            ScalarType::Double => Some(5),
            _ => None,
        }
    }

    let left = match *left {
        ScalarType::UntypedInt => ScalarType::Int,
        ref scalar => scalar.clone(),
    };
    let right = match *right {
        ScalarType::UntypedInt => ScalarType::Int,
        ref scalar => scalar.clone(),
    };

    let left_order = match get_order(&left) {
        Some(order) => order,
        None => panic!("unknown scalar type"),
    };
    let right_order = match get_order(&right) {
        Some(order) => order,
        None => panic!("unknown scalar type"),
    };

    if left_order > right_order {
        left
    } else {
        right
    }
}

fn resolve_arithmetic_types
    (binop: &hst::BinOp,
     left: &ExpressionType,
     right: &ExpressionType)
     -> Result<(ImplicitConversion, ImplicitConversion, hir::Intrinsic2), TyperError> {
    use slp_lang_hir::Type;
    use slp_lang_hir::ScalarType;

    fn common_real_type(left: &ScalarType, right: &ScalarType) -> Result<hir::ScalarType, ()> {
        Ok(most_sig_scalar(left, right))
    }

    // Calculate the output type from the input type and operation
    fn output_type(left: Type, right: Type, op: &hst::BinOp) -> hir::Intrinsic2 {

        // Assert input validity
        {
            let ls = left.0.to_scalar().expect("non-numeric type in binary operation (lhs)");
            let rs = right.0.to_scalar().expect("non-numeric type in binary operation (rhs)");;
            match *op {
                hst::BinOp::LeftShift | hst::BinOp::RightShift | hst::BinOp::BitwiseAnd |
                hst::BinOp::BitwiseOr | hst::BinOp::BitwiseXor => {
                    assert!(ls == ScalarType::Int || ls == ScalarType::UInt,
                            "hir: non-integer source in bitwise op (lhs)");
                    assert!(rs == ScalarType::Int || rs == ScalarType::UInt,
                            "hir: non-integer source in bitwise op (rhs)");
                }
                hst::BinOp::BooleanAnd | hst::BinOp::BooleanOr => {
                    assert!(ls == ScalarType::Bool,
                            "hir: non-boolean source in boolean op (lhs)");
                    assert!(rs == ScalarType::Bool,
                            "hir: non-boolean source in boolean op (rhs)");
                }
                _ => {}
            }
        }

        // Get the more important input type, that serves as the base to
        // calculate the output type from
        let dty = {
            let nd = match most_sig_type_dim(&left.0, &right.0) {
                Some(nd) => nd,
                None => panic!("non-arithmetic numeric type in binary operation"),
            };

            let st = left.0.to_scalar().unwrap();
            assert_eq!(st, right.0.to_scalar().unwrap());
            hir::DataType(hir::DataLayout::new(st, nd), left.1.clone())
        };

        match *op {
            hst::BinOp::Add => hir::Intrinsic2::Add(dty),
            hst::BinOp::Subtract => hir::Intrinsic2::Subtract(dty),
            hst::BinOp::Multiply => hir::Intrinsic2::Multiply(dty),
            hst::BinOp::Divide => hir::Intrinsic2::Divide(dty),
            hst::BinOp::Modulus => hir::Intrinsic2::Modulus(dty),
            hst::BinOp::LeftShift => hir::Intrinsic2::LeftShift(dty),
            hst::BinOp::RightShift => hir::Intrinsic2::RightShift(dty),
            hst::BinOp::BitwiseAnd => hir::Intrinsic2::BitwiseAnd(dty),
            hst::BinOp::BitwiseOr => hir::Intrinsic2::BitwiseOr(dty),
            hst::BinOp::BitwiseXor => hir::Intrinsic2::BitwiseXor(dty),
            hst::BinOp::LessThan => hir::Intrinsic2::LessThan(dty),
            hst::BinOp::LessEqual => hir::Intrinsic2::LessEqual(dty),
            hst::BinOp::GreaterThan => hir::Intrinsic2::GreaterThan(dty),
            hst::BinOp::GreaterEqual => hir::Intrinsic2::GreaterEqual(dty),
            hst::BinOp::Equality => hir::Intrinsic2::Equality(dty),
            hst::BinOp::Inequality => hir::Intrinsic2::Inequality(dty),
            hst::BinOp::BooleanAnd => hir::Intrinsic2::BooleanAnd(dty),
            hst::BinOp::BooleanOr => hir::Intrinsic2::BooleanOr(dty),
            _ => panic!("unexpected binop in resolve_arithmetic_types"),
        }
    }

    fn do_noerror(op: &hst::BinOp,
                  left: &ExpressionType,
                  right: &ExpressionType)
                  -> Result<(ImplicitConversion, ImplicitConversion, hir::Intrinsic2), ()> {
        let &ExpressionType(hir::Type(ref left_l, ref modl), _) = left;
        let &ExpressionType(hir::Type(ref right_l, ref modr), _) = right;
        let (ltl, rtl) = match (left_l, right_l) {
            (&hir::TypeLayout::Scalar(ref ls), &hir::TypeLayout::Scalar(ref rs)) => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = hir::TypeLayout::from_scalar(common_scalar);
                let common_right = common_left.clone();
                (common_left, common_right)
            }
            (&hir::TypeLayout::Scalar(ref ls), &hir::TypeLayout::Vector(ref rs, ref x2)) => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = hir::TypeLayout::from_scalar(common_scalar.clone());
                let common_right = hir::TypeLayout::from_vector(common_scalar, *x2);
                (common_left, common_right)
            }
            (&hir::TypeLayout::Vector(ref ls, ref x1), &hir::TypeLayout::Scalar(ref rs)) => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = hir::TypeLayout::from_vector(common_scalar.clone(), *x1);
                let common_right = hir::TypeLayout::from_scalar(common_scalar);
                (common_left, common_right)
            }
            (&hir::TypeLayout::Vector(ref ls, ref x1),
             &hir::TypeLayout::Vector(ref rs, ref x2)) if x1 == x2 || *x1 == 1 || *x2 == 1 => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = hir::TypeLayout::from_vector(common_scalar.clone(), *x1);
                let common_right = hir::TypeLayout::from_vector(common_scalar, *x2);
                (common_left, common_right)
            }
            (&hir::TypeLayout::Matrix(ref ls, ref x1, ref y1),
             &hir::TypeLayout::Matrix(ref rs, ref x2, ref y2)) if x1 == x2 && y1 == y2 => {
                let common_scalar = try!(common_real_type(ls, rs));
                let common_left = hir::TypeLayout::from_matrix(common_scalar, *x2, *y2);
                let common_right = common_left.clone();
                (common_left, common_right)
            }
            _ => return Err(()),
        };
        let out_mod = hir::TypeModifier {
            is_const: false,
            row_order: hir::RowOrder::Column,
            precise: modl.precise || modr.precise,
            volatile: false,
        };
        let candidate_left = Type(ltl.clone(), out_mod.clone());
        let candidate_right = Type(rtl.clone(), out_mod.clone());
        let output_type = output_type(candidate_left, candidate_right, op);
        let elt = ExpressionType(hir::Type(ltl, out_mod.clone()), hir::ValueType::Rvalue);
        let lc = try!(ImplicitConversion::find(left, &elt));
        let ert = ExpressionType(hir::Type(rtl, out_mod), hir::ValueType::Rvalue);
        let rc = try!(ImplicitConversion::find(right, &ert));
        Ok((lc, rc, output_type))
    }

    match do_noerror(binop, left, right) {
        Ok(res) => Ok(res),
        Err(_) => {
            Err(TyperError::BinaryOperationWrongTypes(binop.clone(),
                                                      left.to_error_type(),
                                                      right.to_error_type()))
        }
    }
}

fn parse_expr_binop(op: &hst::BinOp,
                    lhs: &hst::Expression,
                    rhs: &hst::Expression,
                    context: &ExpressionContext)
                    -> Result<TypedExpression, TyperError> {
    let lhs_texp = try!(parse_expr(lhs, context));
    let rhs_texp = try!(parse_expr(rhs, context));
    let lhs_pt = lhs_texp.to_error_type();
    let rhs_pt = rhs_texp.to_error_type();
    let err_bad_type = Err(TyperError::BinaryOperationWrongTypes(op.clone(), lhs_pt, rhs_pt));
    let (lhs_ir, lhs_type) = match lhs_texp {
        TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
        _ => return err_bad_type,
    };
    let (rhs_ir, rhs_type) = match rhs_texp {
        TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
        _ => return err_bad_type,
    };
    match *op {
        hst::BinOp::Add | hst::BinOp::Subtract | hst::BinOp::Multiply | hst::BinOp::Divide |
        hst::BinOp::Modulus | hst::BinOp::LessThan | hst::BinOp::LessEqual |
        hst::BinOp::GreaterThan | hst::BinOp::GreaterEqual | hst::BinOp::Equality |
        hst::BinOp::Inequality | hst::BinOp::LeftShift | hst::BinOp::RightShift => {
            if *op == hst::BinOp::LeftShift || *op == hst::BinOp::RightShift {
                fn is_integer(ety: &ExpressionType) -> bool {
                    let sty = match (ety.0).0.to_scalar() {
                        Some(sty) => sty,
                        None => return false,
                    };
                    sty == hir::ScalarType::Int || sty == hir::ScalarType::UInt ||
                    sty == hir::ScalarType::UntypedInt
                }
                if !is_integer(&lhs_type) || !is_integer(&rhs_type) {
                    return err_bad_type;
                }
            }
            let types = try!(resolve_arithmetic_types(op, &lhs_type, &rhs_type));
            let (lhs_cast, rhs_cast, output_intrinsic) = types;
            let lhs_final = Box::new(lhs_cast.apply_pel(lhs_ir));
            let rhs_final = Box::new(rhs_cast.apply_pel(rhs_ir));
            let output_type = output_intrinsic.get_return_type();
            let node = pel::Expression::Intrinsic2(output_intrinsic, lhs_final, rhs_final);
            Ok(TypedExpression::Value(node, output_type))
        }
        hst::BinOp::BitwiseAnd | hst::BinOp::BitwiseOr | hst::BinOp::BitwiseXor |
        hst::BinOp::BooleanAnd | hst::BinOp::BooleanOr => {
            let lhs_tyl = &(lhs_type.0).0;
            let rhs_tyl = &(rhs_type.0).0;
            let lhs_mod = &(lhs_type.0).1;
            let rhs_mod = &(rhs_type.0).1;
            let scalar = if *op == hst::BinOp::BooleanAnd || *op == hst::BinOp::BooleanOr {
                hir::ScalarType::Bool
            } else {
                let lhs_scalar = try!(lhs_tyl.to_scalar()
                    .ok_or(TyperError::BinaryOperationNonNumericType));
                let rhs_scalar = try!(rhs_tyl.to_scalar()
                    .ok_or(TyperError::BinaryOperationNonNumericType));
                match (lhs_scalar, rhs_scalar) {
                    (hir::ScalarType::Int, hir::ScalarType::Int) => hir::ScalarType::Int,
                    (hir::ScalarType::Int, hir::ScalarType::UInt) => hir::ScalarType::UInt,
                    (hir::ScalarType::UInt, hir::ScalarType::Int) => hir::ScalarType::UInt,
                    (hir::ScalarType::UInt, hir::ScalarType::UInt) => hir::ScalarType::UInt,
                    (hir::ScalarType::UntypedInt, hir::ScalarType::Int) => hir::ScalarType::Int,
                    (hir::ScalarType::UntypedInt, hir::ScalarType::UInt) => hir::ScalarType::UInt,
                    (hir::ScalarType::Int, hir::ScalarType::UntypedInt) => hir::ScalarType::Int,
                    (hir::ScalarType::UInt, hir::ScalarType::UntypedInt) => hir::ScalarType::UInt,
                    (hir::ScalarType::UntypedInt, hir::ScalarType::UntypedInt) => {
                        hir::ScalarType::UntypedInt
                    }
                    _ => return err_bad_type,
                }
            };
            let x = hir::TypeLayout::max_dim(lhs_tyl.to_x(), rhs_tyl.to_x());
            let y = hir::TypeLayout::max_dim(lhs_tyl.to_y(), rhs_tyl.to_y());
            let tyl = hir::TypeLayout::from_numeric(scalar, x, y);
            let out_mod = hir::TypeModifier {
                is_const: false,
                row_order: hir::RowOrder::Column,
                precise: lhs_mod.precise || rhs_mod.precise,
                volatile: false,
            };
            let ty = hir::Type(tyl, out_mod).to_rvalue();
            let lhs_cast = match ImplicitConversion::find(&lhs_type, &ty) {
                Ok(cast) => cast,
                Err(()) => return err_bad_type,
            };
            let rhs_cast = match ImplicitConversion::find(&rhs_type, &ty) {
                Ok(cast) => cast,
                Err(()) => return err_bad_type,
            };
            assert_eq!(lhs_cast.get_target_type(), rhs_cast.get_target_type());
            let lhs_final = lhs_cast.apply_pel(lhs_ir);
            let rhs_final = rhs_cast.apply_pel(rhs_ir);
            let dty = match rhs_cast.get_target_type().0.into() {
                Some(dty) => dty,
                None => return err_bad_type,
            };
            let i = match *op {
                hst::BinOp::BitwiseAnd => hir::Intrinsic2::BitwiseAnd(dty),
                hst::BinOp::BitwiseOr => hir::Intrinsic2::BitwiseOr(dty),
                hst::BinOp::BitwiseXor => hir::Intrinsic2::BitwiseXor(dty),
                hst::BinOp::BooleanAnd => hir::Intrinsic2::BooleanAnd(dty),
                hst::BinOp::BooleanOr => hir::Intrinsic2::BooleanOr(dty),
                _ => unreachable!(),
            };
            let output_type = i.get_return_type();
            let node = pel::Expression::Intrinsic2(i, Box::new(lhs_final), Box::new(rhs_final));
            Ok(TypedExpression::Value(node, output_type))
        }
        hst::BinOp::Assignment |
        hst::BinOp::SumAssignment |
        hst::BinOp::DifferenceAssignment |
        hst::BinOp::ProductAssignment |
        hst::BinOp::QuotientAssignment |
        hst::BinOp::RemainderAssignment => {
            let required_rtype = match lhs_type.1 {
                hir::ValueType::Lvalue => {
                    ExpressionType(lhs_type.0.clone(), hir::ValueType::Rvalue)
                }
                _ => return Err(TyperError::LvalueRequired),
            };
            match ImplicitConversion::find(&rhs_type, &required_rtype) {
                Ok(rhs_cast) => {
                    let rhs_final = rhs_cast.apply_pel(rhs_ir);
                    let ty = required_rtype.0.clone();
                    let i = match *op {
                        hst::BinOp::Assignment => hir::Intrinsic2::Assignment(ty),
                        hst::BinOp::SumAssignment |
                        hst::BinOp::DifferenceAssignment |
                        hst::BinOp::ProductAssignment |
                        hst::BinOp::QuotientAssignment |
                        hst::BinOp::RemainderAssignment => {
                            // Find data type for assignment
                            let dtyl = match ty.0.into() {
                                Some(dtyl) => dtyl,
                                None => return err_bad_type,
                            };
                            let dty = hir::DataType(dtyl, ty.1);
                            // Make output intrinsic from source op and data type
                            match *op {
                                hst::BinOp::SumAssignment => hir::Intrinsic2::SumAssignment(dty),
                                hst::BinOp::DifferenceAssignment => {
                                    hir::Intrinsic2::DifferenceAssignment(dty)
                                }
                                hst::BinOp::ProductAssignment => {
                                    hir::Intrinsic2::ProductAssignment(dty)
                                }
                                hst::BinOp::QuotientAssignment => {
                                    hir::Intrinsic2::QuotientAssignment(dty)
                                }
                                hst::BinOp::RemainderAssignment => {
                                    hir::Intrinsic2::RemainderAssignment(dty)
                                }
                                _ => unreachable!(),
                            }
                        }
                        _ => unreachable!(),
                    };
                    let output_type = i.get_return_type();
                    let node =
                        pel::Expression::Intrinsic2(i, Box::new(lhs_ir), Box::new(rhs_final));
                    Ok(TypedExpression::Value(node, output_type))
                }
                Err(()) => err_bad_type,
            }
        }
        hst::BinOp::Sequence => return Err(TyperError::ExpressionSequenceOperatorNotImplemented),
    }
}

fn parse_expr_ternary(cond: &hst::Expression,
                      lhs: &hst::Expression,
                      rhs: &hst::Expression,
                      context: &ExpressionContext)
                      -> Result<TypedExpression, TyperError> {
    // Generate sub expressions
    let (cond, cond_ety) = try!(parse_expr_value_only(cond, context));
    let (lhs, lhs_ety) = try!(parse_expr_value_only(lhs, context));
    let (rhs, rhs_ety) = try!(parse_expr_value_only(rhs, context));

    let ExpressionType(lhs_ty, _) = lhs_ety.clone();
    let ExpressionType(rhs_ty, _) = rhs_ety.clone();
    let wrong_types_err = Err(TyperError::TernaryArmsMustHaveSameType(lhs_ty.to_error_type(),
                                                                      rhs_ty.to_error_type()));
    let hir::Type(lhs_tyl, lhs_mod) = lhs_ty;
    let hir::Type(rhs_tyl, rhs_mod) = rhs_ty;

    // Attempt to find best scalar match between match arms
    // This will return None for non-numeric types
    let st = match (lhs_tyl.to_scalar(), rhs_tyl.to_scalar()) {
        (Some(left_scalar), Some(right_scalar)) => {
            Some(most_sig_scalar(&left_scalar, &right_scalar))
        }
        _ => None,
    };

    // Attempt to find best vector match
    // This will return None for non-numeric types
    // This may return None for some combinations of numeric layouts
    let nd = most_sig_type_dim(&lhs_tyl, &rhs_tyl);

    // Transform the types
    let (lhs_target_tyl, rhs_target_tyl) = match (st, nd) {
        (Some(st), Some(nd)) => {
            let dtyl = hir::DataLayout::new(st, nd);
            let tyl = hir::TypeLayout::from_data(dtyl);
            (tyl.clone(), tyl)
        }
        (Some(st), None) => {
            let left = lhs_tyl.transform_scalar(st.clone());
            let right = rhs_tyl.transform_scalar(st);
            (left, right)
        }
        (None, Some(_)) => {
            panic!("internal error: most_sig_scalar failed where most_sig_type_dim succeeded")
        }
        (None, None) => (lhs_tyl, rhs_tyl),
    };

    let comb_tyl = if lhs_target_tyl == rhs_target_tyl {
        lhs_target_tyl
    } else {
        return wrong_types_err;
    };

    let target_mod = hir::TypeModifier {
        is_const: false,
        row_order: lhs_mod.row_order, // TODO: ???
        precise: lhs_mod.precise || rhs_mod.precise,
        volatile: false,
    };

    let ety_target = hir::Type(comb_tyl, target_mod).to_rvalue();

    let left_cast = match ImplicitConversion::find(&lhs_ety, &ety_target) {
        Ok(cast) => cast,
        Err(()) => return wrong_types_err,
    };
    let right_cast = match ImplicitConversion::find(&rhs_ety, &ety_target) {
        Ok(cast) => cast,
        Err(()) => return wrong_types_err,
    };

    let lhs_casted = Box::new(left_cast.apply_pel(lhs));
    let rhs_casted = Box::new(right_cast.apply_pel(rhs));
    assert_eq!(left_cast.get_target_type(), right_cast.get_target_type());
    let final_type = left_cast.get_target_type();

    // Cast the condition
    let cond_cast = match ImplicitConversion::find(&cond_ety, &hir::Type::bool().to_rvalue()) {
        Ok(cast) => cast,
        Err(()) => {
            return Err(TyperError::TernaryConditionRequiresBoolean(cond_ety.to_error_type()))
        }
    };
    let cond_casted = Box::new(cond_cast.apply_pel(cond));

    let node = pel::Expression::TernaryConditional(cond_casted, lhs_casted, rhs_casted);
    Ok(TypedExpression::Value(node, final_type))
}

fn parse_expr_unchecked(ast: &hst::Expression,
                        context: &ExpressionContext)
                        -> Result<TypedExpression, TyperError> {
    match ast {
        &hst::Expression::Literal(ref lit) => parse_literal(lit),
        &hst::Expression::Variable(ref s) => parse_variable(s, context),
        &hst::Expression::UnaryOperation(ref op, ref expr) => parse_expr_unaryop(op, expr, context),
        &hst::Expression::BinaryOperation(ref op, ref lhs, ref rhs) => {
            parse_expr_binop(op, lhs, rhs, context)
        }
        &hst::Expression::TernaryConditional(ref cond, ref lhs, ref rhs) => {
            parse_expr_ternary(cond, lhs, rhs, context)
        }
        &hst::Expression::ArraySubscript(ref array, ref subscript) => {
            let array_texp = try!(parse_expr(array, context));
            let subscript_texp = try!(parse_expr(subscript, context));
            let (array_ir, array_ty) = match array_texp {
                TypedExpression::Value(array_ir, array_ty) => (array_ir, array_ty),
                _ => return Err(TyperError::ArrayIndexingNonArrayType),
            };
            let (subscript_ir, subscript_ty) = match subscript_texp {
                TypedExpression::Value(subscript_ir, subscript_ty) => (subscript_ir, subscript_ty),
                _ => return Err(TyperError::ArrayIndexingNonArrayType),
            };
            let ExpressionType(hir::Type(array_tyl, _), _) = array_ty;
            let node = try!(match array_tyl {
                hir::TypeLayout::Array(_, _) |
                hir::TypeLayout::Object(hir::ObjectType::Buffer(_)) |
                hir::TypeLayout::Object(hir::ObjectType::RWBuffer(_)) |
                hir::TypeLayout::Object(hir::ObjectType::StructuredBuffer(_)) |
                hir::TypeLayout::Object(hir::ObjectType::RWStructuredBuffer(_)) => {
                    let index = hir::Type::int().to_rvalue();
                    let cast_to_int_result = ImplicitConversion::find(&subscript_ty, &index);
                    let subscript_final = match cast_to_int_result {
                        Err(_) => return Err(TyperError::ArraySubscriptIndexNotInteger),
                        Ok(cast) => cast.apply_pel(subscript_ir),
                    };
                    let array = Box::new(array_ir);
                    let sub = Box::new(subscript_final);
                    let sub_node = pel::Expression::ArraySubscript(array, sub);
                    Ok(sub_node)
                }
                hir::TypeLayout::Object(hir::ObjectType::Texture2D(data_type)) => {
                    let index = hir::Type::intn(2).to_rvalue();
                    let cast = ImplicitConversion::find(&subscript_ty, &index);
                    let subscript_final = match cast {
                        Err(_) => return Err(TyperError::ArraySubscriptIndexNotInteger),
                        Ok(cast) => cast.apply_pel(subscript_ir),
                    };
                    let array = Box::new(array_ir);
                    let sub = Box::new(subscript_final);
                    let sub_node = pel::Expression::Texture2DIndex(data_type, array, sub);
                    Ok(sub_node)
                }
                hir::TypeLayout::Object(hir::ObjectType::RWTexture2D(data_type)) => {
                    let index = hir::Type::intn(2).to_rvalue();
                    let cast = ImplicitConversion::find(&subscript_ty, &index);
                    let subscript_final = match cast {
                        Err(_) => return Err(TyperError::ArraySubscriptIndexNotInteger),
                        Ok(cast) => cast.apply_pel(subscript_ir),
                    };
                    let array = Box::new(array_ir);
                    let sub = Box::new(subscript_final);
                    let sub_node = pel::Expression::RWTexture2DIndex(data_type, array, sub);
                    Ok(sub_node)
                }
                _ => Err(TyperError::ArrayIndexingNonArrayType),
            });
            let ety = match pel::TypeParser::get_expression_type(&node,
                                                                 context.as_type_context()) {
                Ok(ety) => ety,
                Err(_) => panic!("internal error: type unknown"),
            };
            Ok(TypedExpression::Value(node, ety))
        }
        &hst::Expression::Member(ref composite, ref member) => {
            let composite_texp = try!(parse_expr(composite, context));
            let composite_pt = composite_texp.to_error_type();
            let (composite_ir, composite_ty) = match composite_texp {
                TypedExpression::Value(composite_ir, composite_type) => {
                    (composite_ir, composite_type)
                }
                _ => return Err(TyperError::TypeDoesNotHaveMembers(composite_pt)),
            };
            let ExpressionType(hir::Type(composite_tyl, composite_mod), vt) = composite_ty;
            match &composite_tyl {
                &hir::TypeLayout::Struct(ref id) => {
                    match context.find_struct_member(id, member) {
                        Ok(ty) => {
                            let composite = Box::new(composite_ir);
                            let member = pel::Expression::Member(composite, member.clone());
                            Ok(TypedExpression::Value(member, ty.to_lvalue()))
                        }
                        Err(err) => Err(err),
                    }
                }
                &hir::TypeLayout::Scalar(_) => {
                    if member == "x" || member == "r" {
                        let composite_ty = hir::Type(composite_tyl.clone(), composite_mod);
                        let ty = ExpressionType(composite_ty, vt);
                        // Just emit the composite expression and drop the member / swizzle
                        return Ok(TypedExpression::Value(composite_ir, ty));
                    }

                    // Scalars don't really have members, so return a sensible error message
                    return Err(TyperError::TypeDoesNotHaveMembers(composite_pt));
                }
                &hir::TypeLayout::Vector(ref scalar, ref x) => {
                    let mut swizzle_slots = Vec::with_capacity(member.len());
                    for c in member.chars() {
                        swizzle_slots.push(match c {
                            'x' | 'r' if *x >= 1 => hir::SwizzleSlot::X,
                            'y' | 'g' if *x >= 2 => hir::SwizzleSlot::Y,
                            'z' | 'b' if *x >= 3 => hir::SwizzleSlot::Z,
                            'w' | 'a' if *x >= 4 => hir::SwizzleSlot::W,
                            _ => {
                                return Err(TyperError::InvalidSwizzle(composite_pt, member.clone()))
                            }
                        });
                    }
                    // Lets say single element swizzles go to scalars
                    // Technically they might be going to 1 element vectors
                    // that then get downcasted
                    // But it's hard to tell as scalars + single element vectors
                    // have the same overload precedence
                    let ty = if swizzle_slots.len() == 1 {
                        hir::TypeLayout::Scalar(scalar.clone())
                    } else {
                        hir::TypeLayout::Vector(scalar.clone(), swizzle_slots.len() as u32)
                    };
                    let ety = ExpressionType(hir::Type(ty, composite_mod), vt);
                    let node = pel::Expression::Swizzle(Box::new(composite_ir), swizzle_slots);
                    Ok(TypedExpression::Value(node, ety))
                }
                &hir::TypeLayout::Object(ref object_type) => {
                    match intrinsics::get_method(object_type, &member) {
                        Ok(intrinsics::MethodDefinition(object_type, name, method_overloads)) => {
                            let overloads = method_overloads.iter()
                                .map(|&(ref param_types, ref factory)| {
                                    let return_type = match *factory {
                                        IntrinsicFactory::Intrinsic0(ref i) => i.get_return_type(),
                                        IntrinsicFactory::Intrinsic1(ref i) => i.get_return_type(),
                                        IntrinsicFactory::Intrinsic2(ref i) => i.get_return_type(),
                                        IntrinsicFactory::Intrinsic3(ref i) => i.get_return_type(),
                                    };
                                    FunctionOverload(FunctionName::Intrinsic(factory.clone()),
                                                     return_type.0,
                                                     param_types.clone())
                                })
                                .collect::<Vec<_>>();
                            Ok(
                                TypedExpression::Method(UnresolvedMethod(
                                    name,
                                    hir::Type::from_object(object_type),
                                    overloads,
                                    composite_ir
                                ))
                            )
                        }
                        Err(()) => Err(TyperError::UnknownTypeMember(composite_pt, member.clone())),
                    }
                }
                // Todo: Matrix components + Object members
                _ => return Err(TyperError::TypeDoesNotHaveMembers(composite_pt)),
            }
        }
        &hst::Expression::Call(ref func, ref params) => {
            let func_texp = try!(parse_expr(func, context));
            let mut params_ir: Vec<pel::Expression> = vec![];
            let mut params_types: Vec<ExpressionType> = vec![];
            for param in params {
                let expr_texp = try!(parse_expr(param, context));
                let (expr_ir, expr_ty) = match expr_texp {
                    TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
                    texp => return Err(TyperError::FunctionPassedToAnotherFunction(
                        func_texp.to_error_type(),
                        texp.to_error_type()
                    )),
                };
                params_ir.push(expr_ir);
                params_types.push(expr_ty);
            }
            match func_texp {
                TypedExpression::Function(unresolved) => {
                    write_function(unresolved, &params_types, params_ir)
                }
                TypedExpression::Method(unresolved) => {
                    write_method(unresolved, &params_types, params_ir)
                }
                _ => return Err(TyperError::CallOnNonFunction),
            }
        }
        &hst::Expression::NumericConstructor(ref dtyl, ref params) => {
            let datalayout = try!(parse_datalayout(dtyl));
            let target_scalar = datalayout.to_scalar();
            let mut slots: Vec<pel::ConstructorSlot> = vec![];
            let mut total_arity = 0;
            for param in params {
                let expr_texp = try!(parse_expr(param, context));
                let (expr_base, ety) = match expr_texp {
                    TypedExpression::Value(expr_ir, expr_ty) => (expr_ir, expr_ty),
                    _ => return Err(TyperError::FunctionNotCalled),
                };
                let &ExpressionType(hir::Type(ref expr_tyl, _), _) = &ety;
                let arity = expr_tyl.get_num_elements();
                total_arity = total_arity + arity;
                let s = target_scalar.clone();
                let target_tyl = match *expr_tyl {
                    hir::TypeLayout::Scalar(_) => hir::TypeLayout::Scalar(s),
                    hir::TypeLayout::Vector(_, ref x) => hir::TypeLayout::Vector(s, *x),
                    hir::TypeLayout::Matrix(_, ref x, ref y) => hir::TypeLayout::Matrix(s, *x, *y),
                    _ => return Err(TyperError::WrongTypeInConstructor),
                };
                let target_type = hir::Type::from_layout(target_tyl).to_rvalue();
                let cast = match ImplicitConversion::find(&ety, &target_type) {
                    Ok(cast) => cast,
                    Err(()) => return Err(TyperError::WrongTypeInConstructor),
                };
                let expr = cast.apply_pel(expr_base);
                slots.push(pel::ConstructorSlot {
                    arity: arity,
                    expr: expr,
                });
            }
            let type_layout = hir::TypeLayout::from_data(datalayout.clone());
            let expected_layout = type_layout.get_num_elements();
            let ty = hir::Type::from_layout(type_layout).to_rvalue();
            if total_arity == expected_layout {
                let cons = pel::Expression::NumericConstructor(datalayout, slots);
                Ok(TypedExpression::Value(cons, ty))
            } else {
                Err(TyperError::NumericConstructorWrongArgumentCount)
            }
        }
        &hst::Expression::Cast(ref ty, ref expr) => {
            let expr_texp = try!(parse_expr(expr, context));
            let expr_pt = expr_texp.to_error_type();
            match expr_texp {
                TypedExpression::Value(expr_ir, _) => {
                    let ir_type = try!(parse_type(ty, context.as_struct_id_finder()));
                    Ok(TypedExpression::Value(pel::Expression::Cast(ir_type.clone(),
                                                                    Box::new(expr_ir)),
                                              ir_type.to_rvalue()))
                }
                _ => Err(TyperError::InvalidCast(expr_pt, ErrorType::Untyped(ty.clone()))),
            }
        }
    }
}

fn parse_expr(expr: &hst::Expression,
              context: &ExpressionContext)
              -> Result<TypedExpression, TyperError> {
    let texp = try!(parse_expr_unchecked(expr, context));
    match texp {
        #[cfg(debug_assertions)]
        TypedExpression::Value(ref expr, ref ty_expected) => {
            let ty_res = pel::TypeParser::get_expression_type(&expr, context.as_type_context());
            let ty = ty_res.expect("type unknown");
            assert!(ty == *ty_expected,
                    "{:?} == {:?}: {:?}",
                    ty,
                    *ty_expected,
                    expr);
        }
        _ => {}
    };
    Ok(texp)
}

pub fn parse_expr_value_only(expr: &hst::Expression,
                             context: &ExpressionContext)
                             -> Result<(pel::Expression, ExpressionType), TyperError> {
    let expr_ir = try!(parse_expr(expr, context));
    match expr_ir {
        TypedExpression::Value(expr, ety) => Ok((expr, ety)),
        TypedExpression::Function(_) => Err(TyperError::FunctionNotCalled),
        TypedExpression::Method(_) => Err(TyperError::FunctionNotCalled),
    }
}
