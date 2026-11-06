# Options pricing and Greeks

A Python implementation of Black–Scholes–Merton for European calls and puts.
It calculates the option price, delta, gamma, theta, vega, and rho. Given an
option price, it can also solve for implied volatility.

The calculations use Python's standard library. You need Python 3.10 or newer.

## Running it

Open a terminal in this folder and run:

```sh
python3 -m option_engine --spot 100 --strike 100 --years 1 --rate 0.05 --vol 0.20
```

This prices a one-year call with a spot and strike of 100, a 5% interest rate,
and 20% volatility. The price is about **10.450584** per unit of underlying.
The output is JSON and includes all five Greeks.

Use `--kind put` to price a put. For a continuous dividend yield, add
`--dividend-yield 0.02`.

To calculate implied volatility from a price of 10.45:

```sh
python3 -m option_engine --spot 100 --strike 100 --years 1 --rate 0.05 --vol 0.20 --market-price 10.45
```

The result is approximately `0.19998445`, or 19.998445%. The price and Greeks
in the output still use the volatility passed through `--vol`. The implied
volatility is reported separately.

You can run everything from this folder without installing the package. If
you prefer an installed command, run `python3 -m pip install -e .` in a virtual
environment, then use `option-engine` instead of `python3 -m option_engine`.

## Using it in Python

```python
from option_engine import OptionInputs, greeks, implied_volatility, price

option = OptionInputs(
    spot=100,
    strike=100,
    maturity=1.0,
    rate=0.05,
    volatility=0.20,
    kind="call",
    dividend_yield=0.0,
)

option_price = price(option)
sensitivities = greeks(option)
implied_vol = implied_volatility(option_price, option)

print(option_price)
print(sensitivities.delta)
print(sensitivities.theta_per_day)
print(implied_vol)
```

`OptionInputs` is immutable. To change one input, use `dataclasses.replace`:

```python
from dataclasses import replace

put = replace(option, kind="put")
print(price(put))
```

## Inputs and units

Use decimals for rates and volatility: 5% is `0.05`, and 20% is `0.20`.
`maturity` is the time remaining in years. For calendar days, `days / 365`
gives an ACT/365 year fraction.

Prices are per unit of underlying, in the same currency as spot and strike.
For a position, multiply the price and Greeks by the signed quantity and the
actual contract multiplier.

| Result | Meaning |
| --- | --- |
| `delta` | Price sensitivity to a one-unit change in spot |
| `gamma` | Delta sensitivity to a one-unit change in spot |
| `theta_per_year` | Price sensitivity to the passage of one year |
| `theta_per_day` | Annual theta divided by 365 |
| `vega_per_unit` | Price derivative with respect to decimal volatility |
| `vega_per_point` | Vega divided by 100 |
| `rho_per_unit` | Price derivative with respect to the decimal interest rate |
| `rho_per_point` | Rho divided by 100 |

A move from 20% to 21% volatility is one percentage point, so use
`vega_per_point` for that estimate. Theta measures time passing, which is the
negative of the derivative with respect to time remaining. All Greeks are
local sensitivities; a large input change needs a fresh price calculation.

## Model assumptions and edge cases

The model assumes European exercise, constant volatility and interest rates,
and a constant continuous dividend yield. It does not handle American early
exercise or discrete dividend payments.

At expiry, `price` returns the payoff. With zero volatility, it returns the
discounted deterministic payoff. `greeks` requires positive time and
volatility; the command-line output uses `null` for Greeks at those boundaries.
Negative rates and dividend yields are accepted.

The implied-volatility solver uses bisection. It ignores the volatility stored
in `OptionInputs` and checks the quote against the model's price bounds first.
At the lower bound, it returns zero. It rejects the upper bound because there
is no finite implied volatility there, and it rejects expired options.

The default price tolerance is `1e-10`, so quotes within that amount above the
lower bound also return zero. The search goes up to `16.0` volatility by
default, or 1600% annually. Tolerances, the iteration limit, and the upper
volatility limit can be changed through the Python API. A small price error
can still imply a large volatility error when vega is close to zero.

Calculations use double-precision floating point. Inputs that overflow or
underflow essential calculations, or produce nonfinite Greeks, raise a
`ValueError`. Very small option values can round to zero.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

The tests compare prices and Greeks with reference values, check Greeks by
bumping the inputs, and compare prices with a separate binomial tree. They
also check put–call parity, the pricing equation, implied-volatility recovery,
expiry, invalid inputs, and command-line behaviour.

The pricing code is in `option_engine/core.py`, the command-line interface is
in `option_engine/__main__.py`, and the checks are in `tests/test_engine.py`.

## Next steps

A binomial pricer with American exercise would be a useful second model.
After that, a delta-hedging simulation could compare hedging error at different
rebalancing frequencies, including trading costs.

## References

- [Kerry Back: The Black–Scholes Formula](https://book.derivative-securities.org/Chapter_BlackScholes.html)
- [Options Industry Council: Volatility and the Greeks](https://www.optionseducation.org/advancedconcepts/volatility-the-greeks)
