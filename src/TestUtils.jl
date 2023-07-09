module TestUtils

function test_complex_fft end
function test_real_fft end

function __init__()
    # Better error message if users forget to load Test
    Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
        if exc.f in (test_real_fft, test_complex_fft)
            print(io, "\nDid you forget to load Test?")
        end
    end
end

end