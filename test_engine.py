from dataclasses import replace
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

from option_engine import OptionInputs, greeks, implied_volatility, price, price_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def binomial_european(option: OptionInputs, steps: int = 1000) -> float:
    time_step = option.maturity / steps
    up = math.exp(option.volatility * math.sqrt(time_step))
    down = 1 / up
    growth = math.exp((option.rate - option.dividend_yield) * time_step)
    probability = (growth - down) / (up - down)
    discount = math.exp(-option.rate * time_step)

    if not 0 <= probability <= 1:
        raise ValueError("invalid binomial probability")

    direction = 1 if option.kind == "call" else -1
    values = [
        max(
            direction * (option.spot * up ** (2 * index - steps) - option.strike),
            0.0,
        )
        for index in range(steps + 1)
    ]

    for width in range(steps, 0, -1):
        values = [
            discount * (
                (1 - probability) * values[index]
                + probability * values[index + 1]
            )
            for index in range(width)
        ]

    return values[0]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.option = OptionInputs(
            spot=100, strike=100, maturity=1, rate=0.05, volatility=0.2
        )

    def assert_close(self, actual, expected, relative=1e-6, absolute=1e-7):
        self.assertTrue(
            math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute),
            f"{actual} != {expected}",
        )

    def test_reference_prices(self):
        self.assertAlmostEqual(price(self.option), 10.450583572185565, places=11)
        put = replace(self.option, kind="put")
        self.assertAlmostEqual(price(put), 5.573526022256971, places=11)

    def test_reference_greeks(self):
        result = greeks(self.option)
        expected = {
            "delta": 0.6368306511756191,
            "gamma": 0.0187620173458469,
            "theta_per_year": -6.414027546438197,
            "vega_per_unit": 37.52403469169379,
            "rho_per_unit": 53.232481545376345,
        }
        for name, value in expected.items():
            with self.subTest(greek=name):
                self.assert_close(getattr(result, name), value, relative=1e-10)

    def test_greeks_against_price_changes(self):
        random_values = random.Random(2026)

        for _ in range(30):
            option = OptionInputs(
                spot=random_values.uniform(80, 120),
                strike=100,
                maturity=random_values.uniform(0.1, 3),
                rate=random_values.uniform(-0.04, 0.10),
                volatility=random_values.uniform(0.15, 0.7),
                kind=random_values.choice(("call", "put")),
                dividend_yield=random_values.uniform(0, 0.08),
            )
            with self.subTest(option=option):
                result = greeks(option)
                bump = option.spot * 1e-4
                price_up = price(replace(option, spot=option.spot + bump))
                price_down = price(replace(option, spot=option.spot - bump))
                numerical_delta = (price_up - price_down) / (2 * bump)
                numerical_gamma = (
                    price_up - 2 * price(option) + price_down
                ) / bump**2

                self.assert_close(result.delta, numerical_delta)
                self.assert_close(result.gamma, numerical_gamma)

                sensitivities = (
                    ("volatility", result.vega_per_unit, 1),
                    ("rate", result.rho_per_unit, 1),
                    ("maturity", result.theta_per_year, -1),
                )
                for field, expected, direction in sensitivities:
                    bump = 1e-5
                    value = getattr(option, field)
                    price_up = price(replace(option, **{field: value + bump}))
                    price_down = price(replace(option, **{field: value - bump}))
                    numerical_greek = direction * (price_up - price_down) / (2 * bump)
                    self.assert_close(expected, numerical_greek)

    def test_put_call_parity_and_bounds(self):
        for maturity in (0.1, 1, 3):
            for rate in (-0.03, 0, 0.07):
                for dividend_yield in (-0.02, 0, 0.04):
                    for spot in (30, 80, 100, 120, 300):
                        call = replace(
                            self.option, spot=spot, maturity=maturity,
                            rate=rate, dividend_yield=dividend_yield,
                        )
                        put = replace(call, kind="put")
                        discounted_spot = spot * math.exp(-dividend_yield * maturity)
                        discounted_strike = call.strike * math.exp(-rate * maturity)

                        with self.subTest(option=call):
                            self.assert_close(
                                price(call) - price(put),
                                discounted_spot - discounted_strike,
                            )
                            self.assert_close(
                                greeks(call).delta - greeks(put).delta,
                                math.exp(-dividend_yield * maturity),
                            )
                            self.assert_close(greeks(call).gamma, greeks(put).gamma)
                            self.assert_close(
                                greeks(call).vega_per_unit, greeks(put).vega_per_unit
                            )

                            for option in (call, put):
                                lower, upper = price_bounds(option)
                                self.assertGreaterEqual(price(option), lower - 1e-12)
                                self.assertLessEqual(price(option), upper + 1e-12)

    def test_black_scholes_equation(self):
        for kind in ("call", "put"):
            for rate in (-0.03, 0.07):
                option = replace(
                    self.option, kind=kind, rate=rate,
                    maturity=1.5, dividend_yield=0.03,
                )
                result = greeks(option)
                residual = (
                    result.theta_per_year
                    + (rate - option.dividend_yield) * option.spot * result.delta
                    + 0.5 * option.volatility**2 * option.spot**2 * result.gamma
                    - rate * price(option)
                )
                with self.subTest(option=option):
                    self.assertAlmostEqual(residual, 0, places=10)

    def test_binomial_prices(self):
        for kind in ("call", "put"):
            for dividend_yield in (0, 0.03):
                option = replace(
                    self.option, kind=kind, dividend_yield=dividend_yield
                )
                with self.subTest(option=option):
                    self.assertAlmostEqual(
                        price(option), binomial_european(option), delta=0.003
                    )

    def test_expiry_payoffs(self):
        cases = (
            ("call", 110, 10), ("put", 90, 10),
            ("call", 90, 0), ("put", 110, 0),
            ("call", 100, 0), ("put", 100, 0),
        )
        for kind, spot, payoff in cases:
            option = replace(self.option, spot=spot, maturity=0, kind=kind)
            with self.subTest(option=option):
                self.assertEqual(price(option), payoff)
                self.assertEqual(price_bounds(option), (payoff, payoff))
                with self.assertRaises(ValueError):
                    greeks(option)
                with self.assertRaises(ValueError):
                    implied_volatility(payoff, option)

    def test_zero_volatility_prices(self):
        for kind in ("call", "put"):
            for rate in (-0.05, 0, 0.05):
                option = replace(
                    self.option, kind=kind, rate=rate,
                    volatility=0, dividend_yield=0.02,
                )
                discounted_spot = option.spot * math.exp(-0.02)
                discounted_strike = option.strike * math.exp(-rate)
                difference = discounted_spot - discounted_strike
                expected = max(difference if kind == "call" else -difference, 0)
                with self.subTest(option=option):
                    self.assert_close(price(option), expected)
                    with self.assertRaises(ValueError):
                        greeks(option)

    def test_near_expiry(self):
        for kind in ("call", "put"):
            option = replace(self.option, maturity=1e-10, kind=kind)
            self.assertGreater(price(option), 0)
            self.assertTrue(all(math.isfinite(value) for value in vars(greeks(option)).values()))

    def test_near_identical_spot_and_strike(self):
        option = replace(
            self.option, spot=math.nextafter(100.0, math.inf),
            volatility=1e-16, rate=0,
        )
        with localcontext() as context:
            context.prec = 60
            log_moneyness = (
                Decimal.from_float(option.spot) / Decimal(option.strike)
            ).ln()
            d1 = float(log_moneyness / Decimal.from_float(option.volatility))

        expected_delta = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        self.assertAlmostEqual(greeks(option).delta, expected_delta, places=13)

    def test_implied_volatility_recovery(self):
        for kind in ("call", "put"):
            for volatility in (0.1, 0.2, 0.8, 2.0, 5.0):
                for rate in (-0.02, 0.05):
                    option = replace(
                        self.option, kind=kind, volatility=volatility,
                        rate=rate, dividend_yield=0.03,
                    )
                    market_price = price(option)
                    estimate = implied_volatility(
                        market_price, replace(option, volatility=0)
                    )
                    with self.subTest(option=option):
                        self.assertAlmostEqual(estimate, volatility, delta=1e-8)
                        self.assertAlmostEqual(
                            price(replace(option, volatility=estimate)),
                            market_price, delta=1e-10,
                        )

    def test_implied_volatility_price_bounds(self):
        for kind in ("call", "put"):
            option = replace(self.option, kind=kind)
            lower, upper = price_bounds(option)
            self.assertEqual(implied_volatility(lower, option), 0)
            self.assertEqual(implied_volatility(lower + 5e-11, option), 0)

            for quote in (-1, lower - 1e-5, upper, upper + 1):
                with self.subTest(kind=kind, quote=quote):
                    with self.assertRaises(ValueError):
                        implied_volatility(quote, option)

    def test_implied_volatility_limits(self):
        high_vol_price = price(replace(self.option, volatility=2))
        with self.assertRaisesRegex(ValueError, "exceeds max_volatility"):
            implied_volatility(high_vol_price, self.option, max_volatility=0.5)
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            implied_volatility(price(self.option), self.option, max_iterations=1)
        with self.assertRaisesRegex(RuntimeError, "bracket exhausted"):
            implied_volatility(
                price(self.option), self.option,
                price_tolerance=1e-15, volatility_tolerance=0.5,
            )

    def test_invalid_solver_settings(self):
        cases = (
            {"price_tolerance": 0}, {"price_tolerance": float("nan")},
            {"volatility_tolerance": -1}, {"max_volatility": 0},
            {"max_volatility": float("inf")}, {"max_iterations": True},
            {"max_iterations": 2.5}, {"max_iterations": 0},
        )
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                implied_volatility(price(self.option), self.option, **settings)

        for quote in (float("nan"), float("inf"), True, "10", 10**400):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                implied_volatility(quote, self.option)

    def test_greek_units(self):
        result = greeks(self.option)
        self.assertAlmostEqual(result.theta_per_day * 365, result.theta_per_year)
        self.assertAlmostEqual(result.vega_per_point * 100, result.vega_per_unit)
        self.assertAlmostEqual(result.rho_per_point * 100, result.rho_per_unit)

    def test_invalid_inputs(self):
        fields = ("spot", "strike", "maturity", "rate", "volatility", "dividend_yield")
        for field in fields:
            for value in (True, None, "100", float("nan"), float("inf"), 10**400):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(self.option, **{field: value})

        cases = (
            {"spot": 0}, {"spot": -1}, {"strike": 0}, {"strike": -1},
            {"maturity": -1}, {"volatility": -1}, {"kind": "CALL"}, {"kind": None},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.option, **changes)

    def test_numerical_range_errors(self):
        tiny_volatility = replace(self.option, volatility=5e-324, rate=0, maturity=0.01)
        for calculation in (price, greeks):
            with self.subTest(calculation=calculation.__name__):
                with self.assertRaisesRegex(ValueError, "total volatility"):
                    calculation(tiny_volatility)

        tiny_spot = replace(self.option, spot=1e-310, strike=1e-310, rate=0)
        self.assertTrue(math.isfinite(price(tiny_spot)))
        with self.assertRaisesRegex(ValueError, "Greeks exceed"):
            greeks(tiny_spot)

        for changes in ({"rate": -1000}, {"rate": 1000}, {"dividend_yield": -1000}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                price(replace(self.option, **changes))


class CommandLineTests(unittest.TestCase):
    def run_engine(self, *extra_arguments):
        command = [
            sys.executable, "-B", "-m", "option_engine",
            "--spot", "100", "--strike", "100", "--years", "1",
            "--rate", "0.05", "--vol", "0.2", *extra_arguments,
        ]
        return subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True
        )

    def test_price_and_greeks(self):
        process = self.run_engine()
        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        self.assertAlmostEqual(output["price"], 10.450583572185565, places=11)
        self.assertIn("theta_per_day", output["greeks"])

    def test_put_with_dividends_and_market_price(self):
        process = self.run_engine(
            "--kind", "put", "--dividend-yield", "0.02", "--market-price", "7"
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        option = OptionInputs(100, 100, 1, 0.05, output["implied_volatility"], "put", 0.02)
        self.assertAlmostEqual(price(option), 7, delta=1e-10)
        self.assertEqual(output["inputs"]["volatility"], 0.2)

    def test_expiry_and_zero_volatility(self):
        for arguments in (("--years", "0"), ("--vol", "0")):
            process = self.run_engine(*arguments)
            self.assertEqual(process.returncode, 0, process.stderr)
            output = json.loads(process.stdout)
            self.assertIsNone(output["greeks"])
            self.assertIn("greeks_note", output)

    def test_errors_do_not_print_tracebacks(self):
        cases = (
            ("--market-price", "-1"),
            ("--vol", "nan"),
            ("--vol", "5e-324", "--years", "0.01"),
            ("--spot", "1e-310", "--strike", "1e-310", "--rate", "0"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                process = self.run_engine(*arguments)
                self.assertEqual(process.returncode, 2)
                self.assertIn("error:", process.stderr)
                self.assertNotIn("Traceback", process.stderr)
                self.assertEqual(process.stdout, "")

    def test_help(self):
        process = self.run_engine("--help")
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("0.20 for 20%", process.stdout)


if __name__ == "__main__":
    unittest.main()
