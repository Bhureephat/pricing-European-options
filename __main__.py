import argparse
import json
from dataclasses import asdict

from .core import OptionInputs, greeks, implied_volatility, price


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Price a European option and calculate its Greeks per unit of underlying."
    )
    parser.add_argument("--kind", choices=("call", "put"), default="call")
    parser.add_argument("--spot", type=float, required=True)
    parser.add_argument("--strike", type=float, required=True)
    parser.add_argument("--years", type=float, required=True, help="years until expiry")
    parser.add_argument(
        "--rate", type=float, required=True,
        help="annual continuously compounded rate, such as 0.05 for 5%%",
    )
    parser.add_argument(
        "--vol", type=float, required=True,
        help="annual volatility, such as 0.20 for 20%%",
    )
    parser.add_argument(
        "--dividend-yield", type=float, default=0.0,
        help="annual continuous dividend yield, such as 0.02 for 2%%",
    )
    parser.add_argument(
        "--market-price", type=float,
        help="also solve implied volatility for this option price",
    )
    args = parser.parse_args(argv)

    try:
        option = OptionInputs(
            spot=args.spot,
            strike=args.strike,
            maturity=args.years,
            rate=args.rate,
            volatility=args.vol,
            kind=args.kind,
            dividend_yield=args.dividend_yield,
        )
        output = {"inputs": asdict(option), "price": price(option)}

        if option.maturity > 0 and option.volatility > 0:
            sensitivities = greeks(option)
            output["greeks"] = {
                **asdict(sensitivities),
                "theta_per_day": sensitivities.theta_per_day,
                "vega_per_point": sensitivities.vega_per_point,
                "rho_per_point": sensitivities.rho_per_point,
            }
        else:
            output["greeks"] = None
            output["greeks_note"] = "Greeks require positive maturity and volatility."

        if args.market_price is not None:
            output["implied_volatility"] = implied_volatility(args.market_price, option)

        result = json.dumps(output, indent=2, allow_nan=False)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))

    print(result)


if __name__ == "__main__":
    main()
