from io import StringIO
import requests
import pandas as pd


coordinates = pd.read_excel("coord.xlsx", sheet_name="coord2")

url = "https://daymet.ornl.gov/single-pixel/api/data"

def daymet_to_df(text):
    lines = text.split("\n")
    header_idx = next(i for i, line in enumerate(lines) if line.startswith("year,yday"))
    return pd.read_csv(StringIO(text), skiprows=header_idx)

result = []
for i in coordinates.itertuples(index=False):
    params = {
        "lat": i.LATITUDE,
        "lon": i.LONGITUDE,
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
data.to_csv("daymet2.csv")