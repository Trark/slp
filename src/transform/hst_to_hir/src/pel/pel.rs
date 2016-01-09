
//! # Pel: Parsed Expression Language
//!
//! Minilanguage for expression trees before being reduced
//! Almost identical to the final hir expressions but with different
//! global memory operations

use slp_lang_hir as hir;
use slp_lang_hir::DataLayout;
use slp_lang_hir::DataType;
use slp_lang_hir::ObjectType;
use slp_lang_hir::TypeLayout;
use slp_lang_hir::Type;
use slp_lang_hir::Literal;
use slp_lang_hir::VariableRef;
use slp_lang_hir::GlobalId;
use slp_lang_hir::ConstantBufferId;
use slp_lang_hir::FunctionId;
use slp_lang_hir::SwizzleSlot;
use slp_lang_hir::Intrinsic;
use slp_lang_hir::Intrinsic0;
use slp_lang_hir::Intrinsic1;
use slp_lang_hir::Intrinsic2;
use slp_lang_hir::Intrinsic3;
use slp_lang_hir::ExpressionType;
use slp_lang_hir::ToExpressionType;
use slp_lang_hir::TypeError;
use slp_lang_hir::TypeContext;

/// Error for attempting to convert directly from pel to hir.
///
/// This is not expected to work for all expressions, and should fall back to
/// the rel parser.
pub enum DirectToHirError {
    /// The expression may contain a swizzle to a value thats taken as an lvalue
    NonInputSwizzle,
    /// The expression contains a texture index operation
    TextureIndex,
    /// The expression may contain a cast to a value thats taken as an lvalue
    NonInputCast,
}

#[derive(PartialEq, Debug, Clone)]
pub struct ConstructorSlot {
    pub arity: u32,
    pub expr: Expression,
}

impl ConstructorSlot {
    pub fn direct_to_hir(&self) -> Result<hir::ConstructorSlot, DirectToHirError> {
        Ok(hir::ConstructorSlot {
            arity: self.arity,
            expr: try!(self.expr.direct_to_hir_recur(RequiredVt::Rvt)),
        })
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Variable(VariableRef),
    Global(GlobalId),
    ConstantVariable(ConstantBufferId, String),
    TernaryConditional(Box<Expression>, Box<Expression>, Box<Expression>),
    Swizzle(Box<Expression>, Vec<SwizzleSlot>),
    ArraySubscript(Box<Expression>, Box<Expression>),
    Texture2DIndex(DataType, Box<Expression>, Box<Expression>),
    RWTexture2DIndex(DataType, Box<Expression>, Box<Expression>),
    Member(Box<Expression>, String),
    Call(FunctionId, Vec<Expression>),
    NumericConstructor(DataLayout, Vec<ConstructorSlot>),
    Cast(Type, Box<Expression>),
    Intrinsic0(Intrinsic0),
    Intrinsic1(Intrinsic1, Box<Expression>),
    Intrinsic2(Intrinsic2, Box<Expression>, Box<Expression>),
    Intrinsic3(Intrinsic3, Box<Expression>, Box<Expression>, Box<Expression>),
}

#[derive(PartialEq, Debug, Clone, Copy)]
enum RequiredVt {
    Lvt, // May be required to be lvalue
    Rvt, // Only required to be rvalue
}

impl From<hir::InputModifier> for RequiredVt {
    fn from(im: hir::InputModifier) -> RequiredVt {
        match im {
            hir::InputModifier::In => RequiredVt::Rvt,
            hir::InputModifier::Out | hir::InputModifier::InOut => RequiredVt::Lvt,
        }
    }
}

impl Expression {
    pub fn direct_to_hir_recur(&self, vt: RequiredVt) -> Result<hir::Expression, DirectToHirError> {
        match *self {
            Expression::Literal(ref lit) => Ok(hir::Expression::Literal(lit.clone())),
            Expression::Variable(ref var) => Ok(hir::Expression::Variable(var.clone())),
            Expression::Global(ref id) => Ok(hir::Expression::Global(id.clone())),
            Expression::ConstantVariable(ref id, ref name) => {
                Ok(hir::Expression::ConstantVariable(id.clone(), name.clone()))
            }
            Expression::TernaryConditional(ref cond, ref lhs, ref rhs) => {
                let cond_hir = Box::new(try!(cond.direct_to_hir_recur(RequiredVt::Rvt)));
                let lhs_hir = Box::new(try!(lhs.direct_to_hir_recur(RequiredVt::Rvt)));
                let rhs_hir = Box::new(try!(rhs.direct_to_hir_recur(RequiredVt::Rvt)));
                Ok(hir::Expression::TernaryConditional(cond_hir, lhs_hir, rhs_hir))
            }
            Expression::Swizzle(ref composite, ref swizzle) => {
                if vt == RequiredVt::Rvt {
                    let composite_hir = Box::new(try!(composite.direct_to_hir_recur(vt)));
                    Ok(hir::Expression::Swizzle(composite_hir, swizzle.clone()))
                } else {
                    Err(DirectToHirError::NonInputSwizzle)
                }
            }
            Expression::ArraySubscript(ref arr, ref index) => {
                let arr_hir = Box::new(try!(arr.direct_to_hir_recur(vt)));
                let index_hir = Box::new(try!(index.direct_to_hir_recur(RequiredVt::Rvt)));
                Ok(hir::Expression::ArraySubscript(arr_hir, index_hir))
            }
            Expression::Texture2DIndex(_, _, _) |
            Expression::RWTexture2DIndex(_, _, _) => {
                // Else we're complex and can't handle this
                Err(DirectToHirError::TextureIndex)
            }
            Expression::Member(ref composite, ref name) => {
                let composite_hir = Box::new(try!(composite.direct_to_hir_recur(vt)));
                Ok(hir::Expression::Member(composite_hir, name.clone()))
            }
            Expression::Call(ref id, ref args) => {
                let mut args_hir = Vec::with_capacity(args.len());
                for arg in args {
                    let arg_hir = try!(arg.direct_to_hir_recur(RequiredVt::Lvt));
                    args_hir.push(arg_hir);
                }
                Ok(hir::Expression::Call(id.clone(), args_hir))
            }
            Expression::NumericConstructor(ref dtyl, ref cons) => {
                let mut cons_hir = Vec::with_capacity(cons.len());
                for con in cons {
                    let con_hir = try!(con.direct_to_hir());
                    cons_hir.push(con_hir);
                }
                Ok(hir::Expression::NumericConstructor(dtyl.clone(), cons_hir))
            }
            Expression::Cast(ref ty, ref expr) => {
                if vt == RequiredVt::Rvt {
                    let expr_hir = Box::new(try!(expr.direct_to_hir_recur(vt)));
                    Ok(hir::Expression::Cast(ty.clone(), expr_hir))
                } else {
                    Err(DirectToHirError::NonInputCast)
                }
            }
            Expression::Intrinsic0(ref i) => Ok(hir::Expression::Intrinsic0(i.clone())),
            Expression::Intrinsic1(ref i, ref e1) => {
                let p1_vt = i.get_param1_input_modifier().into();
                let e1_hir = Box::new(try!(e1.direct_to_hir_recur(p1_vt)));
                Ok(hir::Expression::Intrinsic1(i.clone(), e1_hir))
            }
            Expression::Intrinsic2(ref i, ref e1, ref e2) => {
                let p1_vt = i.get_param1_input_modifier().into();
                let e1_hir = Box::new(try!(e1.direct_to_hir_recur(p1_vt)));
                let p2_vt = i.get_param2_input_modifier().into();
                let e2_hir = Box::new(try!(e2.direct_to_hir_recur(p2_vt)));
                Ok(hir::Expression::Intrinsic2(i.clone(), e1_hir, e2_hir))
            }
            Expression::Intrinsic3(ref i, ref e1, ref e2, ref e3) => {
                let p1_vt = i.get_param1_input_modifier().into();
                let e1_hir = Box::new(try!(e1.direct_to_hir_recur(p1_vt)));
                let p2_vt = i.get_param2_input_modifier().into();
                let e2_hir = Box::new(try!(e2.direct_to_hir_recur(p2_vt)));
                let p3_vt = i.get_param3_input_modifier().into();
                let e3_hir = Box::new(try!(e3.direct_to_hir_recur(p3_vt)));
                Ok(hir::Expression::Intrinsic3(i.clone(), e1_hir, e2_hir, e3_hir))
            }
        }
    }

    /// Translate directly from pel to hir
    /// This will fail for expression that can't be trivially represented
    pub fn direct_to_hir(&self) -> Result<hir::Expression, DirectToHirError> {
        self.direct_to_hir_recur(RequiredVt::Rvt)
    }

    /// Maybe change the expression based on not expecting the return result
    /// of the expression from being read
    pub fn ignore_return(self) -> Expression {
        // Rewrite common way of writing textures as texture store intrinsic
        if let &Expression::Intrinsic2(Intrinsic2::Assignment(_), ref e1, ref e2) = &self {
            if let Expression::RWTexture2DIndex(ref dty, ref tex, ref index) = **e1 {
                let new_i = Intrinsic3::RWTexture2DStore(dty.clone());
                return Expression::Intrinsic3(new_i, tex.clone(), index.clone(), e2.clone());
            }
        }
        self
    }
}

pub struct TypeParser;

impl TypeParser {
    pub fn get_expression_type(expression: &Expression,
                               context: &TypeContext)
                               -> Result<ExpressionType, TypeError> {
        match *expression {
            Expression::Literal(ref lit) => Ok(hir::TypeParser::get_literal_type(lit)),
            Expression::Variable(ref var_ref) => context.get_local(var_ref),
            Expression::Global(ref id) => context.get_global(id),
            Expression::ConstantVariable(ref id, ref name) => context.get_constant(id, name),
            Expression::TernaryConditional(_, ref expr_left, ref expr_right) => {
                // Ensure the layouts of each side are the same
                // Value types + modifiers can be different
                assert_eq!((try!(TypeParser::get_expression_type(expr_left, context)).0).0,
                           (try!(TypeParser::get_expression_type(expr_right, context)).0).0);
                let ety = try!(TypeParser::get_expression_type(expr_left, context));
                Ok(ety.0.to_rvalue())
            }
            Expression::Swizzle(ref vec, ref swizzle) => {
                let ExpressionType(Type(vec_tyl, vec_mod), vec_vt) =
                    try!(TypeParser::get_expression_type(vec, context));
                let tyl = match vec_tyl {
                    TypeLayout::Vector(ref scalar, _) => {
                        if swizzle.len() == 1 {
                            TypeLayout::Scalar(scalar.clone())
                        } else {
                            TypeLayout::Vector(scalar.clone(), swizzle.len() as u32)
                        }
                    }
                    _ => return Err(TypeError::InvalidTypeForSwizzle(vec_tyl.clone())),
                };
                Ok(ExpressionType(Type(tyl, vec_mod), vec_vt))
            }
            Expression::ArraySubscript(ref array, _) => {
                let array_ty = try!(TypeParser::get_expression_type(&array, context));
                // Todo: Modifiers on object type template parameters
                Ok(match (array_ty.0).0 {
                    TypeLayout::Array(ref element, _) => {
                        Type::from_layout(*element.clone()).to_lvalue()
                    }
                    TypeLayout::Object(ObjectType::Buffer(data_type)) => {
                        Type::from_data(data_type).to_lvalue()
                    }
                    TypeLayout::Object(ObjectType::RWBuffer(data_type)) => {
                        Type::from_data(data_type).to_lvalue()
                    }
                    TypeLayout::Object(ObjectType::StructuredBuffer(structured_type)) => {
                        Type::from_structured(structured_type).to_lvalue()
                    }
                    TypeLayout::Object(ObjectType::RWStructuredBuffer(structured_type)) => {
                        Type::from_structured(structured_type).to_lvalue()
                    }
                    tyl => return Err(TypeError::ArrayIndexMustBeUsedOnArrayType(tyl)),
                })
            }
            Expression::Texture2DIndex(ref data_type, _, _) => {
                Ok(Type::from_data(data_type.clone()).to_lvalue())
            }
            Expression::RWTexture2DIndex(ref data_type, _, _) => {
                Ok(Type::from_data(data_type.clone()).to_lvalue())
            }
            Expression::Member(ref expr, ref name) => {
                let expr_type = try!(TypeParser::get_expression_type(&expr, context));
                let id = match (expr_type.0).0 {
                    TypeLayout::Struct(id) => id,
                    tyl => {
                        return Err(TypeError::MemberNodeMustBeUsedOnStruct(tyl.clone(),
                                                                           name.clone()))
                    }
                };
                context.get_struct_member(&id, name)
            }
            Expression::Call(ref id, _) => context.get_function_return(id),
            Expression::NumericConstructor(ref dtyl, _) => {
                Ok(Type::from_layout(TypeLayout::from_data(dtyl.clone())).to_rvalue())
            }
            Expression::Cast(ref ty, _) => Ok(ty.to_rvalue()),
            Expression::Intrinsic0(ref intrinsic) => Ok(intrinsic.get_return_type()),
            Expression::Intrinsic1(ref intrinsic, _) => Ok(intrinsic.get_return_type()),
            Expression::Intrinsic2(ref intrinsic, _, _) => Ok(intrinsic.get_return_type()),
            Expression::Intrinsic3(ref intrinsic, _, _, _) => Ok(intrinsic.get_return_type()),
        }
    }
}
