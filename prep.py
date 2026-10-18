from io import StringIO
import requests
import pandas as pd
import argparse
import os
import sys

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="001_RSESQ_niveaux_eau_2026-08-04.xlsx", type=str)
    parser.add_argument("--sheet", default="Metadonnees_puits", type=str)
    parser.add_argument("--features", default=["ID_PUITS", "LATITUDE", "LONGITUDE"], nargs="+")
    parser.add_argument("--save", default="daymet.csv", type=str)
    return parser.parse_args(args)


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    coordinates = pd.read_excel(args.path, sheet_name=args.sheet)
    coordinates = coordinates[args.features]

    url = "https://daymet.ornl.gov/single-pixel/api/data"

    def daymet_to_df(text):
        lines = text.split("\n")
        header_idx = next(i for i, line in enumerate(lines) if line.startswith("year,yday"))
        return pd.read_csv(StringIO(text), skiprows=header_idx)

    result = []
    for i in coordinates.itertuples(index=False):
        params = {
            "lat": i.LATITUDE, #45.8782355735536, # 49.08858
            "lon": i.LONGITUDE, # -73.1290083455576 # -66.6684
            "vars": "tmax,tmin,prcp,srad,vp,swe,dayl"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()

        df = daymet_to_df(response.text)
        df.columns = df.columns.str.strip()
        df["lat"] = i.LATITUDE
        df["lon"] = i.LONGITUDE
        df["ID"] = i.ID_PUITS
        result.append(df)

    data = pd.concat(result, ignore_index=True)
    directory = os.makedirs("data", exist_ok=True)
    data.to_csv(args.save)