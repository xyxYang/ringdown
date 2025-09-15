import pytest
import ringdown.qnms
import ringdown.result

T_MSUN = 4.9254909476412675e-06
M_REF = 70
C_REF = 0.7
FTAU_REF = {
    (1, -2, 2, 2, 0): (245.85210273356427, 0.004267509656122605),
    (1, -2, 2, 2, 1): (240.57155728864586, 0.001411671895902423),
    (1, -2, 3, 2, 0): (350.440511991165, 0.004095329813033091)
}
QUADRATIC_FTAU_REF = {
    (2, 2, 0, 2, 2, 0): (491.70420546712853, 0.0021337548280613025),
    (2, 2, 0, 3, 2, 0): (596.2926147247292, 0.0020898236282765746)
}


def test_T_MSUN():
    assert ringdown.qnms.T_MSUN == T_MSUN


@pytest.mark.parametrize("index, f_tau_ref", FTAU_REF.items())
def test_get_ftau(index, f_tau_ref):
    (p, s, l, m, n) = index
    ftau = ringdown.qnms.get_ftau(M_REF, C_REF, n=n, l=l, m=p*m)
    assert ftau == pytest.approx(f_tau_ref, rel=1e-12)


class TestKerrMode:

    def setup_method(self, method):
        self.mass = M_REF
        self.spin = C_REF
        self.kerr_modes = {m: ringdown.qnms.KerrMode(*m)
                           for m in FTAU_REF.keys()}

    def test_ftau(self):
        for (p, s, l, m, n), mode in self.kerr_modes.items():
            ref = ringdown.qnms.get_ftau(self.mass, self.spin, n=n, l=l, m=p*m)
            assert mode.ftau(self.spin, self.mass) == ref

    def test_fgamma(self):
        for (p, s, l, m, n), mode in self.kerr_modes.items():
            f_ref, t_ref = ringdown.qnms.get_ftau(self.mass, self.spin, n=n,
                                                  l=l, m=p*m)
            f, g = mode.fgamma(self.spin, self.mass)
            assert (f, g) == pytest.approx((f_ref, 1/t_ref), abs=1e-12)

    @pytest.mark.slow
    def test_ftau_approx(self):
        for (p, s, l, m, n), mode in self.kerr_modes.items():
            ref = ringdown.qnms.get_ftau(self.mass, self.spin, n=n, l=l, m=p*m)
            ftau = mode.ftau(self.spin, self.mass, approx=True)
            assert ftau == pytest.approx(ref, rel=1E-2)


class TestQuadraticMode:

    def setup_method(self, method):
        self.mass = M_REF
        self.spin = C_REF
        self.quadratic_modes = {m: ringdown.qnms.QuadraticMode(*m)
                                for m in QUADRATIC_FTAU_REF.keys()}

    def test_ftau(self):
        for (l1, m1, n1, l2, m2, n2), mode in self.quadratic_modes.items():
            f_ref1, t_ref1 = ringdown.qnms.get_ftau(self.mass, self.spin, n=n1, l=l1, m=m1)
            f_ref2, t_ref2 = ringdown.qnms.get_ftau(self.mass, self.spin, n=n2, l=l2, m=m2)
            f_ref = f_ref1 + f_ref2
            t_ref = 1 / (1 / t_ref1 + 1 / t_ref2)
            f, t = mode.ftau(self.spin, self.mass)
            assert (f, t) == pytest.approx((f_ref, t_ref), abs=1e-12)

    def test_fgamma(self):
        for (l1, m1, n1, l2, m2, n2), mode in self.quadratic_modes.items():
            f_ref1, t_ref1 = ringdown.qnms.get_ftau(self.mass, self.spin, n=n1, l=l1, m=m1)
            f_ref2, t_ref2 = ringdown.qnms.get_ftau(self.mass, self.spin, n=n2, l=l2, m=m2)
            f_ref = f_ref1 + f_ref2
            t_ref = 1 / (1 / t_ref1 + 1 / t_ref2)
            f, g = mode.fgamma(self.spin, self.mass)
            assert (f, g) == pytest.approx((f_ref, 1/t_ref), abs=1e-12)

    @pytest.mark.slow
    def test_ftau_approx(self):
        for (l1, m1, n1, l2, m2, n2), mode in self.quadratic_modes.items():
            f_ref1, t_ref1 = ringdown.qnms.get_ftau(self.mass, self.spin, n=n1, l=l1, m=m1)
            f_ref2, t_ref2 = ringdown.qnms.get_ftau(self.mass, self.spin, n=n2, l=l2, m=m2)
            f_ref = f_ref1 + f_ref2
            t_ref = 1 / (1 / t_ref1 + 1 / t_ref2)
            ref = (f_ref, t_ref)
            ftau = mode.ftau(self.spin, self.mass, approx=True)
            assert ftau == pytest.approx(ref, rel=1E-2)


def test_get_parameter_label_map():
    ringdown.qnms.get_parameter_label_map(
        pars=ringdown.result._DATAFRAME_PARAMETERS,
        modes=FTAU_REF.keys())


@pytest.mark.parametrize("parameter", ringdown.result._DATAFRAME_PARAMETERS)
def test_parameter_label(parameter):
    p = ringdown.qnms.ParameterLabel(parameter)
    assert isinstance(p.get_label(latex=True), str)
    assert isinstance(p.get_label(latex=False), str)
    assert isinstance(p.get_key(), str)


if __name__ == "__main__":
    pytest.main()
