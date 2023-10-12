import BDIO: BDIO_write!, BDIO_read

function BDIO.BDIO_write!(fb::BDIO.BDIOstream, u1ws::U1)
    for mu in 1:2, iy in 1:u1ws.params.iL[2], ix in 1:u1ws.params.iL[1]
        BDIO.BDIO_write!(fb,[real(u1ws.U[ix,iy,mu])])
        BDIO.BDIO_write!(fb,[imag(u1ws.U[ix,iy,mu])])
    end
end

function BDIO.BDIO_read(fb::BDIO.BDIOstream, u1ws::U1)
    reg = zeros(Float64,2)
    for mu in 1:2, iy in 1:u1ws.params.iL[2], ix in 1:u1ws.params.iL[1]
        BDIO.BDIO_read(fb,reg)
        u1ws.U[ix,iy,mu] = complex(reg[1], reg[2])
    end
end
