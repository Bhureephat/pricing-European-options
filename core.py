from dataclasses import dataclass, replace
from math import erfc, exp, isfinite, log, log1p, pi, sqrt
from typing import Literal


def _validate_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")

    try:
        finite = isfinite(value)
    except OverflowError:
        finite = False

    if not finite:
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class OptionInputs:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    kind: Literal["call", "put"] = "call"
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "spot", "strike", "maturity", "rate", "volatility", "dividend_yield"
        ):
            _validate_number(name, getattr(self, name))

        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("spot and strike must be positive")
        if self.maturity < 0 or self.volatility < 0:
            raise ValueError("maturity and volatility must be nonnegative")
        if self.kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta_per_year: float
    vega_per_unit: float
    rho_per_unit: float

    @property
    def theta_per_day(self) -> float:
        return self.theta_per_year / 365

    @property
    def vega_per_point(self) -> float:
        return self.vega_per_unit / 100

    @property
    def rho_per_point(self) -> float:
        return self.rho_per_unit / 100


def _normal_cdf(value: float) -> float:
    return 0.5 * erfc(-value / sqrt(2))


def _payoff(spot: float, strike: float, kind: str) -> float:
    if kind == "call":
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _discounted_prices(option: OptionInputs) -> tuple[float, float]:
    try:
        discounted_spot = option.spot * exp(-option.dividend_yield * option.maturity)
        discounted_strike = option.strike * exp(-option.rate * option.maturity)
    except OverflowError as error:
        raise ValueError("discounted prices exceed floating-point range") from error

    if not all(
        isfinite(value) and value > 0
        for value in (discounted_spot, discounted_strike)
    ):
        raise ValueError("discounted prices are outside floating-point range")

    return discounted_spot, discounted_strike


def _d_values(option: OptionInputs) -> tuple[float, float]:
    total_volatility = option.volatility * sqrt(option.maturity)
    if not isfinite(total_volatility) or total_volatility <= 0:
        raise ValueError("total volatility is outside floating-point range")

    relative_difference = (option.spot - option.strike) / option.strike
    if abs(relative_difference) < 0.5:
        log_moneyness = log1p(relative_difference)
    else:
        log_moneyness = log(option.spot) - log(option.strike)

    carry = (option.rate - option.dividend_yield) * option.maturity
    forward_moneyness = log_moneyness + carry
    if not isfinite(forward_moneyness):
        raise ValueError("forward moneyness is outside floating-point range")

    d1 = forward_moneyness / total_volatility + total_volatility / 2
    d2 = d1 - total_volatility
    return d1, d2


def price_bounds(option: OptionInputs) -> tuple[float, float]:
    if option.maturity == 0:
        intrinsic_value = _payoff(option.spot, option.strike, option.kind)
        return intrinsic_value, intrinsic_value

    discounted_spot, discounted_strike = _discounted_prices(option)
    lower_bound = _payoff(discounted_spot, discounted_strike, option.kind)
    upper_bound = discounted_spot if option.kind == "call" else discounted_strike
    return lower_bound, upper_bound


def price(option: OptionInputs) -> float:
    if option.maturity == 0:
        return _payoff(option.spot, option.strike, option.kind)

    discounted_spot, discounted_strike = _discounted_prices(option)
    if option.volatility == 0:
        return _payoff(discounted_spot, discounted_strike, option.kind)

    d1, d2 = _d_values(option)
    if discounted_spot <= discounted_strike:
        call_price = max(
            discounted_spot * _normal_cdf(d1)
            - discounted_strike * _normal_cdf(d2),
            0.0,
        )
        if option.kind == "call":
            return call_price
        return call_price + (discounted_strike - discounted_spot)

    put_price = max(
        discounted_strike * _normal_cdf(-d2)
        - discounted_spot * _normal_cdf(-d1),
        0.0,
    )
    if option.kind == "put":
        return put_price
    return put_price + (discounted_spot - discounted_strike)


def greeks(option: OptionInputs) -> Greeks:
    if option.maturity == 0 or option.volatility == 0:
        raise ValueError("Greeks require positive maturity and volatility")

    discounted_spot, discounted_strike = _discounted_prices(option)
    d1, d2 = _d_values(option)
    root_time = sqrt(option.maturity)
    total_volatility = option.volatility * root_time
    dividend_discount = exp(-option.dividend_yield * option.maturity)
    density = exp(-0.5 * d1 * d1) / sqrt(2 * pi)
    time_decay = -0.5 * discounted_spot * density * option.volatility / root_time

    if option.kind == "call":
        delta = dividend_discount * _normal_cdf(d1)
        theta = (
            time_decay
            + option.dividend_yield * discounted_spot * _normal_cdf(d1)
            - option.rate * discounted_strike * _normal_cdf(d2)
        )
        rho = option.maturity * (discounted_strike * _normal_cdf(d2))
    else:
        delta = -dividend_discount * _normal_cdf(-d1)
        theta = (
            time_decay
            - option.dividend_yield * discounted_spot * _normal_cdf(-d1)
            + option.rate * discounted_strike * _normal_cdf(-d2)
        )
        rho = -option.maturity * (discounted_strike * _normal_cdf(-d2))

    result = Greeks(
        delta=delta,
        gamma=dividend_discount * density / option.spot / total_volatility,
        theta_per_year=theta,
        vega_per_unit=discounted_spot * density * root_time,
        rho_per_unit=rho,
    )
    if not all(isfinite(value) for value in vars(result).values()):
        raise ValueError("Greeks exceed floating-point range")

    return result


def implied_volatility(
    market_price: float,
    option: OptionInputs,
    *,
    price_tolerance: float = 1e-10,
    volatility_tolerance: float = 1e-12,
    max_iterations: int = 200,
    max_volatility: float = 16.0,
) -> float:
    for name, value in (
        ("market_price", market_price),
        ("price_tolerance", price_tolerance),
        ("volatility_tolerance", volatility_tolerance),
        ("max_volatility", max_volatility),
    ):
        _validate_number(name, value)

    if price_tolerance <= 0 or volatility_tolerance <= 0 or max_volatility <= 0:
        raise ValueError("tolerances and max_volatility must be positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    if option.maturity == 0:
        raise ValueError("implied volatility cannot be determined at expiry")

    lower_price, upper_price = price_bounds(option)
    if market_price < lower_price or market_price >= upper_price:
        raise ValueError(f"market_price must be in [{lower_price}, {upper_price})")
    if market_price - lower_price <= price_tolerance:
        return 0.0

    lower_volatility = 0.0
    upper_volatility = min(0.5, max_volatility)
    upper_quote = price(replace(option, volatility=upper_volatility))

    while upper_quote < market_price and upper_volatility < max_volatility:
        upper_volatility = min(upper_volatility * 2, max_volatility)
        upper_quote = price(replace(option, volatility=upper_volatility))

    if upper_quote < market_price:
        raise ValueError("implied volatility exceeds max_volatility; increase the bracket")

    for _ in range(max_iterations):
        estimate = (lower_volatility + upper_volatility) / 2
        estimate_price = price(replace(option, volatility=estimate))
        error = estimate_price - market_price

        if abs(error) <= price_tolerance:
            return estimate
        if upper_volatility - lower_volatility <= volatility_tolerance:
            raise RuntimeError("volatility bracket exhausted before price tolerance was met")

        if error < 0:
            lower_volatility = estimate
        else:
            upper_volatility = estimate

    raise RuntimeError("implied volatility did not converge; increase max_iterations")
