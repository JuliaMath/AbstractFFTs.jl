module AbstractFFTsEnzymeCoreExt

using AbstractFFTs
using AbstractFFTs.LinearAlgebra
using EnzymeCore
using EnzymeCore.EnzymeRules

######################
# Forward-mode rules #
######################

const DuplicatedOrBatchDuplicated{T} = Union{Duplicated{T},BatchDuplicated{T}}

# since FFTs are linear, implement all forward-model rules generically at a low-level

function EnzymeRules.forward(
    func::Const{typeof(mul!)},
    RT::Type{<:Const},
    y::DuplicatedOrBatchDuplicated{<:StridedArray{T}},
    p::Const{<:AbstractFFTs.Plan{T}},
    x::DuplicatedOrBatchDuplicated{<:StridedArray{T}},
) where {T}
    val = func.val(y.val, p.val, x.val)
    if x isa Duplicated && y isa Duplicated
        dval = func.val(y.dval, p.val, x.dval)
    elseif x isa Duplicated && y isa Duplicated
        dval = map(y.dval, x.dval) do dy, dx
            return func.val(dy, p.val, dx)
        end
    end
    return nothing
end

function EnzymeRules.forward(
    func::Const{typeof(*)},
    RT::Type{
        <:Union{Const,Duplicated,DuplicatedNoNeed,BatchDuplicated,BatchDuplicatedNoNeed}
    },
    p::Const{<:AbstractFFTs.Plan},
    x::DuplicatedOrBatchDuplicated{<:StridedArray},
)
    RT <: Const && return func.val(p.val, x.val)
    if x isa Duplicated
        dval = func.val(p.val, x.dval)
        RT <: DuplicatedNoNeed && return dval
        val = func.val(p.val, x.val)
        RT <: Duplicated && return Duplicated(val, dval)
    else  # x isa BatchDuplicated
        dval = map(x.dval) do dx
            return func.val(p.val, dx)
        end
        RT <: BatchDuplicatedNoNeed && return dval
        val = func.val(p.val, x.val)
        RT <: BatchDuplicated && return BatchDuplicated(val, dval)
    end
end

end  # module
