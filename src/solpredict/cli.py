from __future__ import annotations

import argparse
import json
import sys

from . import config, data


def main(argv=None):
    ap = argparse.ArgumentParser(prog="solpredict",
                                 description="Day-ahead solar irradiance forecasting for Ibri, Oman.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import", help="combine NASA POWER CSV files saved from a browser into data/")
    p.add_argument("files", nargs="+")
    sub.add_parser("download", help="re-fetch the data from the NASA POWER API (internet needed)")
    sub.add_parser("verify", help="check data/ibri_hourly.csv.gz against data/manifest.json")
    sub.add_parser("check", help="print data sanity checks")
    sub.add_parser("run", help="train, choose a model on 2023, test on 2024-2025, write results/")
    sub.add_parser("audit", help="run the verification checks behind docs/verification.md")
    p = sub.add_parser("predict", help="forecast the day after the last full day in a NASA POWER CSV")
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    args = ap.parse_args(argv)

    if args.cmd in ("import", "download"):
        df = data.import_files(args.files) if args.cmd == "import" else data.download()
        m = data.save(df)
        print(f"saved {m['rows']:,} hours ({m['first_hour']} to {m['last_hour']}) to {data.DATA_FILE.name}")
        print(json.dumps(data.check(df), indent=2))
    elif args.cmd == "verify":
        problems = data.verify()
        if problems:
            print("data check FAILED:\n  " + "\n  ".join(problems))
            sys.exit(1)
        print("data/ibri_hourly.csv.gz matches data/manifest.json")
    elif args.cmd == "check":
        print(json.dumps(data.check(data.load()), indent=2))
    elif args.cmd == "run":
        from .evaluate import run
        run(data.load())
        print(f"wrote {config.RESULTS_DIR.name}/ and {config.MODELS_DIR.name}/")
    elif args.cmd == "audit":
        from .audit import run_audit
        out = run_audit(data.load())
        print(json.dumps(out, indent=2))
    elif args.cmd == "predict":
        from .predict import forecast_file
        out = forecast_file(args.input, args.output)
        print(out.to_string(index=False))
        day = out["ghi_forecast_wm2"].sum() / 1000
        print(f"\n{out['date'].iloc[0]}: {day:.2f} kWh/m² of sunlight expected; "
              f"peak plant output {out['plant_mw_forecast'].max():.0f} MW of {config.PLANT_MW:.0f} MW")


if __name__ == "__main__":
    main()
