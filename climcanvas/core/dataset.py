# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Masato Mori
# This file is part of ClimCanvas, licensed under the GNU Affero General Public
# License, version 3 only; see LICENSE and LICENSE.exception.
"""netCDFファイルの読み込みとメタデータの自動認識。"""

from __future__ import annotations

import json
import warnings

import numpy as np
import xarray as xr

_TIME_NAMES = {"time", "times", "date", "year"}
_LAT_NAMES = {"lat", "latitude", "ylat", "nav_lat"}
_LON_NAMES = {"lon", "longitude", "xlon", "nav_lon"}
_VERTICAL_NAMES = {
    "lev", "level", "levels", "plev", "pres", "pressure",
    "height", "altitude", "depth", "isobaric",
}
_LAT_UNITS = {"degrees_north", "degree_north", "degrees_n", "degreen"}
_LON_UNITS = {"degrees_east", "degree_east", "degrees_e", "degreee"}
_VERTICAL_UNITS = {"hpa", "pa", "millibars", "mb", "hectopascals", "m", "km"}


# datetime64 に無損失で変換できる cftime 暦 (日付集合が proleptic_gregorian の
# 部分集合になるもの)
_CONVERTIBLE_CALENDARS = {"noleap", "365_day", "standard", "gregorian",
                           "proleptic_gregorian"}

# 1年の長さが固定の暦 (存在しない日付を含み datetime64 に無損失変換できない)。
# これらは「開始年からの通し日数」の数値軸に変換する (1年目 = 1..360 など)
_FIXED_YEAR_LENGTH_CALENDARS = {"360_day": 360.0, "all_leap": 366.0,
                                 "366_day": 366.0}

# 変換したことを記録する dataset attrs キー。値は "座標名:元の暦:変換方式"
# (方式は datetime64 / daynum)。scriptgen が生成スクリプトに同じ変換行を
# 出すために参照する (1対1対応)
CALENDAR_CONVERTED_ATTR = "_climcanvas_converted_calendar"

# 時刻役割の手動上書き用の dataset attrs キー。値 = 時刻として扱う座標名
# (lag 相関の lag 軸など、自動認識で時刻にならない数値次元を時刻扱いにする)。
# UI 層が読み込み後に assign_attrs で付与し、detect_coord_roles が最優先で読む。
# ファイルには書かれないメモリ上だけの目印 (calendar 変換の attr と同じ設計)
ROLE_TIME_OVERRIDE_ATTR = "_climcanvas_role_time"

# 座標ファイル (経緯度が別ファイルにあるデータ — ClimCORE の FLON.nc / FLAT.nc 等) を
# 結び付けたことを記録する dataset attrs キー。値は JSON 文字列
# `[[絶対パス, 変数名], ...]` (経度, 緯度の順)。scriptgen が生成スクリプトに同じ
# open + assign_coords 行を出すために参照する (1対1対応)。暦変換の attr と同じ
# メモリ上だけの目印
COORD_FILES_ATTR = "_climcanvas_coord_files"


def _convert_cftime_calendar(ds: xr.Dataset) -> xr.Dataset:
    """cftime の時間座標を matplotlib で描ける型に変換する。

    matplotlib は cftime オブジェクトを直接描画できない (nc-time-axis が必要)
    ため、
    - 無損失で変換できる暦 (noleap 等) は datetime64 (proleptic_gregorian) へ
    - 固定長暦 (360_day / all_leap) は開始年からの**通し日数**の数値軸へ
      (daily なら 1年目 = 1..360、2年目 = 361..720。サブデイリーは小数)
    - 変換可能な暦でも日付が datetime64[ns] の表現範囲 (約1678〜2262年) の
      外にあるもの (ダミー時刻の 0000 年・古気候データ等) は、
      先頭時刻からの**経過日数**の数値軸へ
    変換する。変換した座標名・元の暦・方式を attrs に記録する。
    """
    for name in list(ds.coords):
        coord = ds.coords[name]
        if coord.ndim != 1 or coord.dtype != object or coord.size == 0:
            continue
        calendar = getattr(coord.values[0], "calendar", None)
        if calendar in _CONVERTIBLE_CALENDARS:
            try:
                ds = ds.convert_calendar("proleptic_gregorian", dim=str(name),
                                          use_cftime=False)
            except ValueError:
                # 日付が datetime64[ns] の範囲外 (「暦の種類」は変換可能でも
                # 「値の範囲」で失敗する)。経過日数の数値軸にフォールバック。
                # 式は scriptgen._open_dataset_lines の elapsed 分岐と1対1対応
                t0 = coord.values[0]
                days = [d.days + d.seconds / 86400
                        for d in (t - t0 for t in coord.values)]
                ds = ds.assign_coords({name: days})
                ds[name].attrs.update({
                    "units": "day",
                    "long_name": f"time (days since {t0.isoformat()}, "
                                 f"{calendar} calendar)",
                })
                ds.attrs[CALENDAR_CONVERTED_ATTR] = f"{name}:{calendar}:elapsed"
                continue
            ds.attrs[CALENDAR_CONVERTED_ATTR] = f"{name}:{calendar}:datetime64"
        elif calendar in _FIXED_YEAR_LENGTH_CALENDARS:
            length = _FIXED_YEAR_LENGTH_CALENDARS[calendar]
            y0 = int(coord.values[0].year)
            days = [
                (t.year - y0) * length + t.dayofyr
                + t.hour / 24 + t.minute / 1440 + t.second / 86400
                for t in coord.values
            ]
            ds = ds.assign_coords({name: days})
            ds[name].attrs.update({
                "units": "day",
                "long_name": f"time (day number, {calendar} calendar)",
            })
            ds.attrs[CALENDAR_CONVERTED_ATTR] = f"{name}:{calendar}:daynum"
    return ds


def open_dataset(path: str) -> xr.Dataset:
    with warnings.catch_warnings():
        # datetime64 範囲外の時刻軸 (0000年等) で毎回出る定型警告。
        # cftime への劣化は _convert_cftime_calendar が処理し、変換した事実は
        # attrs (CALENDAR_CONVERTED_ATTR / long_name) に残るので抑制してよい
        warnings.filterwarnings(
            "ignore", message="Unable to decode time axis",
            category=xr.SerializationWarning)
        # decode_timedelta=False: timedelta 風 units ("days" 等) の変数 (ラグ相関の
        # lag 座標など) を timedelta64 にデコードさせず、生の数値 + units 属性の
        # まま読む。timedelta64 は matplotlib で描けず UI の .sel も通らない。
        # scriptgen._open_dataset_lines の open 行と1対1対応
        return _convert_cftime_calendar(
            xr.open_dataset(path, decode_timedelta=False))


def open_coord_file(path: str) -> xr.Dataset:
    """座標ファイル (経緯度だけのファイル) を開く。scriptgen の open 行と1対1対応。"""
    return xr.open_dataset(path, decode_timedelta=False)


def attach_coord_files(ds: xr.Dataset,
                       coord_files: list[tuple[str, xr.Dataset]]) -> xr.Dataset:
    """別ファイルの経緯度変数を ds の座標として結び付けた新しい Dataset を返す。

    coord_files は [(パス, 開いた Dataset), ...]。1 ファイルに lon/lat が両方あっても、
    経度と緯度が別ファイル (ClimCORE の FLON.nc / FLAT.nc) でもよい。各ファイルの変数を
    _coord_role (units / standard_name / 名前) で判定し、最初に見つかった lon と lat を
    使う。変数の dims が ds の dims に含まれ長さが一致することを検査し、
    値は位置ベースで付ける (座標ファイル側の x/y 座標値による整列は行わない —
    格子番号が 0 始まりと 1 始まりで食い違っても結び付く)。変数名はファイルのまま。

    ds に既に lat/lon 役割の座標があれば何もせず ds をそのまま返す (誤指定の安全側。
    呼び出し側は attrs の COORD_FILES_ATTR の有無で結び付けたかを判定できる)。
    lon/lat が見つからない・次元が合わないときは ValueError (英語メッセージ。
    UI 側でそのまま表示する)。
    """
    roles = detect_coord_roles(ds)
    if roles["lat"] and roles["lon"]:
        return ds
    found: dict[str, tuple[str, str, xr.DataArray]] = {}
    for path, cds in coord_files:
        for name, var in cds.variables.items():
            if var.ndim < 1:
                continue
            role = _coord_role(str(name), cds[name])
            if role in ("lon", "lat") and role not in found:
                found[role] = (path, str(name), cds[name])
    missing = [r for r in ("lon", "lat") if r not in found]
    if missing:
        raise ValueError(
            f"No {' / '.join(missing)} variable (units degrees_east / degrees_north, "
            f"or a name containing lon / lat) found in the coordinate file(s): "
            f"{', '.join(p for p, _ in coord_files)}")
    new_coords = {}
    for role in ("lon", "lat"):
        path, name, var = found[role]
        for d in var.dims:
            if d not in ds.dims:
                raise ValueError(
                    f"Coordinate variable {name!r} in {path} has dimension {d!r}, "
                    f"which the dataset does not have (dataset dims: "
                    f"{', '.join(str(x) for x in ds.dims)})")
            if ds.sizes[d] != var.sizes[d]:
                raise ValueError(
                    f"Coordinate variable {name!r} in {path} has {var.sizes[d]} "
                    f"points along {d!r}, but the dataset has {ds.sizes[d]}")
        new_coords[name] = (tuple(str(d) for d in var.dims), var.values, dict(var.attrs))
    record = json.dumps([[found[r][0], found[r][1]] for r in ("lon", "lat")])
    return ds.assign_coords(new_coords).assign_attrs({COORD_FILES_ATTR: record})


def detect_coord_roles(ds: xr.Dataset) -> dict[str, str | None]:
    """座標の役割 (lat / lon / vertical / time) を属性と名前から自動判定する。

    attrs に ROLE_TIME_OVERRIDE_ATTR があれば、その座標を最優先で時刻役割に
    割り当てる (自動判定より優先。存在しない座標名なら無視して自動判定)。
    上書きされた座標は他の役割の自動判定からも除外する。
    """
    roles: dict[str, str | None] = {"lat": None, "lon": None, "vertical": None, "time": None}
    override = str(ds.attrs.get(ROLE_TIME_OVERRIDE_ATTR, "") or "")
    if override and override in ds.coords:
        roles["time"] = override
    for name in ds.coords:
        if str(name) == roles["time"]:
            continue
        role = _coord_role(str(name), ds.coords[name])
        if role is not None and roles[role] is None:
            roles[role] = str(name)
    return roles


def _coord_role(name: str, coord: xr.DataArray) -> str | None:
    units = str(coord.attrs.get("units", "")).lower()
    standard_name = str(coord.attrs.get("standard_name", "")).lower()
    axis = str(coord.attrs.get("axis", "")).upper()
    lname = name.lower()

    if np.issubdtype(coord.dtype, np.datetime64) or standard_name == "time" \
            or axis == "T" or lname in _TIME_NAMES:
        return "time"
    if standard_name == "latitude" or units in _LAT_UNITS or axis == "Y" or lname in _LAT_NAMES:
        return "lat"
    if standard_name == "longitude" or units in _LON_UNITS or axis == "X" or lname in _LON_NAMES:
        return "lon"
    if axis == "Z" or "positive" in coord.attrs or lname in _VERTICAL_NAMES \
            or (units in _VERTICAL_UNITS and lname not in _LAT_NAMES | _LON_NAMES):
        return "vertical"
    return None


def is_curvilinear(ds: xr.Dataset, roles: dict[str, str | None] | None = None) -> bool:
    """lat/lon 座標が 2 次元 (curvilinear 格子: lat(y,x) / lon(y,x)) かを返す。

    海洋モデル・WRF・ClimCORE のランベルト格子など、経緯度が格子番号 (y, x) の
    関数として 2 次元変数で与えられるデータ。lat と lon が同じ 2 つの dim を
    同じ順で持つときだけ True (片方だけ 2 次元、dims が食い違うものは対象外)。
    render / scriptgen / UI の「2 次元座標の分岐」は必ずこの判定を通すこと —
    1 次元格子は既存経路をそのまま通る (図・スクリプトの見た目互換)。
    """
    roles = roles or detect_coord_roles(ds)
    lat, lon = roles.get("lat"), roles.get("lon")
    if lat is None or lon is None:
        return False
    la, lo = ds[lat], ds[lon]
    return la.ndim == 2 and lo.ndim == 2 and tuple(la.dims) == tuple(lo.dims)


def horizontal_dims(ds: xr.Dataset,
                    roles: dict[str, str | None] | None = None) -> tuple[str, str] | None:
    """水平面を張る 2 つの dim 名を (lat 側, lon 側) の順で返す。

    「lat/lon の**座標名**」と「水平面の **dim 名**」を区別するための入口。
    1 次元格子では両者が一致する (`(roles["lat"], roles["lon"])` をそのまま返す。
    地点データのように lat/lon が dim でない 1 次元補助座標のときも同じ —
    呼び出し側の `dim in var.dims` 判定が従来どおり偽になる)。curvilinear 格子では
    lat 座標の dims (例 `("y", "x")`) を返す。lat / lon のどちらかが無ければ None。
    """
    roles = roles or detect_coord_roles(ds)
    lat, lon = roles.get("lat"), roles.get("lon")
    if lat is None or lon is None:
        return None
    if is_curvilinear(ds, roles):
        return tuple(str(d) for d in ds[lat].dims)  # type: ignore[return-value]
    return (lat, lon)


def lonlat_bounds(ds: xr.Dataset,
                  roles: dict[str, str | None] | None = None) -> dict | None:
    """データの経緯度範囲を返す (水平面図の領域指定の初期値用)。

    返り値は {"lon_min", "lon_max", "lat_min", "lat_max", "lon_global", "global"}。
    lon/lat は 1 次元 (格子・地点) でも 2 次元 (curvilinear。全格子点を含む矩形) でも
    よく、NaN は無視する。経度が全周を覆う (1 次元: 範囲 + 格子間隔 ≥ 360°、
    2 次元: 範囲 ≥ 360°) ときは lon_global = True とし、経度範囲をその規約の全周
    (0–360 または -180–180) に置き換える (357.5 までの格子をそのまま領域にすると
    357.5–360 の隙間が空くため)。global は経度が全周かつ緯度が両極近くまで
    (範囲 170° 以上。ガウス格子の ±88° 等も全球扱い) あること。
    lat / lon の役割が無い、または有限値が無いときは None。
    """
    roles = roles or detect_coord_roles(ds)
    lat, lon = roles.get("lat"), roles.get("lon")
    if lat is None or lon is None:
        return None
    lon_v = np.asarray(ds[lon].values, dtype=float)
    lat_v = np.asarray(ds[lat].values, dtype=float)
    if lon_v.size == 0 or not np.isfinite(lon_v).any() or not np.isfinite(lat_v).any():
        return None
    raw_min = float(np.nanmin(lon_v))
    lat_min, lat_max = float(np.nanmin(lat_v)), float(np.nanmax(lat_v))
    # 経度は円周上の量なので、単純な min/max ではなく「全点を含む最小の弧」を取る
    # (日付変更線や 0° をまたぐ格子 — -180 規約で 140〜180 と -180〜-140 に分かれた
    # 1 次元格子や、ランベルト格子の高緯度側が 180° を越える 2 次元座標 — で min/max
    # が全周に化けるのを防ぐ)。[0, 360) に畳んで並べ、最大の隙間の反対側が求める弧。
    # 全周の判定も隙間で行う: 最大の隙間が「格子間隔 (隙間の中央値) の 1.5 倍か 5° の
    # 大きい方」以下なら全周 (0〜357.5 の 2.5° 格子は wrap の隙間 2.5° で全周)
    u = np.unique(np.mod(lon_v[np.isfinite(lon_v)].ravel(), 360.0))
    if u.size == 1:
        a = b = float(u[0])
        max_gap, step = 360.0, 360.0
    else:
        gaps = np.diff(u)
        wrap_gap = float(u[0] + 360.0 - u[-1])
        k = int(np.argmax(gaps))
        if wrap_gap >= gaps[k]:
            a, b, max_gap = float(u[0]), float(u[-1]), wrap_gap
        else:
            a, b, max_gap = float(u[k + 1]), float(u[k]) + 360.0, float(gaps[k])
        step = float(np.median(gaps))
    lon_global = max_gap <= max(1.5 * step, 5.0)
    if lon_global:
        lo = 0.0 if raw_min >= 0.0 else -180.0
        lon_min, lon_max = lo, lo + 360.0
    else:
        # 元が -180〜180 規約なら弧もその規約に寄せる (330–390 ではなく -30–30)
        if raw_min < 0.0 and a >= 180.0:
            a -= 360.0
            b -= 360.0
        lon_min, lon_max = a, b
    return {"lon_min": lon_min, "lon_max": lon_max,
            "lat_min": lat_min, "lat_max": lat_max,
            "lon_global": lon_global,
            "global": lon_global and (lat_max - lat_min) >= 170.0}


# 円錐定数 n から標準緯線の対を復元するときに優先する「よく使われる対」。
# 同じ n を与える標準緯線の対は無数にあり (n = ln(cos φ1 / cos φ2) / ln(tan(π/4+φ2/2) /
# tan(π/4+φ1/2))、1 本なら n = sin φ)、n が同じなら図の形は同じ (一様な縮尺だけが違う)。
# 表示用には既知の対 (JMA の 30/60、cartopy 既定の 33/45 など) を優先し、どれにも
# 合わなければ 1 本 (φ = asin n) を 2 つ並べる
_KNOWN_STANDARD_PARALLELS = [(30.0, 60.0), (33.0, 45.0), (25.0, 25.0), (40.0, 40.0),
                             (45.0, 45.0), (50.0, 50.0), (60.0, 60.0)]


def _cone_constant(lat1: float, lat2: float) -> float:
    """ランベルト正角円錐図法の円錐定数 n (球面)。lat1 == lat2 なら sin(lat1)。"""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    if abs(lat1 - lat2) < 1e-9:
        return float(np.sin(p1))
    return float(np.log(np.cos(p1) / np.cos(p2))
                 / np.log(np.tan(np.pi / 4 + p2 / 2) / np.tan(np.pi / 4 + p1 / 2)))


def _lcc_normalized_xy(lon2d, lat2d, lon0: float, n: float):
    """球面ランベルト正角円錐図法の正規化座標 (縮尺 F = 1、原点は任意)。"""
    theta = n * np.deg2rad(((lon2d - lon0 + 180.0) % 360.0) - 180.0)
    rho = np.tan(np.pi / 4 - np.deg2rad(lat2d) / 2) ** n
    return rho * np.sin(theta), -rho * np.cos(theta)


def _lcc_grid_misfit(lon2d, lat2d, lon0: float, n: float) -> float:
    """(lon0, n) で投影したとき格子が「x は列だけ・y は行だけの関数」からどれだけ外れるか
    (格子間隔の 2 乗で正規化した相対残差)。0 に近いほど直交等間隔格子。"""
    x, y = _lcc_normalized_xy(lon2d, lat2d, lon0, n)
    xc = x.mean(axis=0)
    yc = y.mean(axis=1)
    res = float(((x - xc[None, :]) ** 2).mean() + ((y - yc[:, None]) ** 2).mean())
    sp2 = float((np.diff(xc) ** 2).mean() + (np.diff(yc) ** 2).mean())
    return res / sp2 if sp2 > 0.0 else np.inf


def detect_grid_projection(ds: xr.Dataset,
                           roles: dict[str, str | None] | None = None) -> dict | None:
    """2 次元座標格子の投影法を経緯度から数値的に推定する (図の初期投影法用)。

    座標ファイルには投影の属性が無いことが多い (ClimCORE の FLON / FLAT 等) ので、
    「その投影で投影すると格子が直交等間隔になる」ことを当てはめて判定する。対象は
    ランベルト正角円錐図法 (球面) の族: 中心経度 lon0 と円錐定数 n を粗い格子探索 →
    3 段階の細分化で求め、相対残差が閾値未満で、かつ投影した格子の間隔が一様なら採用。
    n が 1 に近ければ極ステレオ図法 (n = 1 が極ステレオの極限) として返す。
    格子の x 軸が投影の x 軸と平行 (回転していない) ことを仮定する。

    返り値は panel["projection"] に入れられる dict (投影名・中心経度・中心緯度・
    標準緯線) に "cone_constant" / "misfit" を添えたもの。curvilinear でない、
    または当てはまらなければ None。行・列を最大 25 点ずつ間引いて評価するので
    数百 ms 以内 (実測 0.24 秒、817×661)。
    """
    roles = roles or detect_coord_roles(ds)
    if not is_curvilinear(ds, roles):
        return None
    lon_full = np.asarray(ds[roles["lon"]].values, dtype=float)
    lat_full = np.asarray(ds[roles["lat"]].values, dtype=float)
    ny, nx = lon_full.shape
    if ny < 4 or nx < 4:
        return None
    ii = np.unique(np.linspace(0, ny - 1, min(25, ny)).astype(int))
    jj = np.unique(np.linspace(0, nx - 1, min(25, nx)).astype(int))
    lon2d = lon_full[np.ix_(ii, jj)]
    lat2d = lat_full[np.ix_(ii, jj)]
    if not (np.isfinite(lon2d).all() and np.isfinite(lat2d).all()):
        return None
    lon_c = float(lon2d[lon2d.shape[0] // 2, lon2d.shape[1] // 2])

    best = (np.inf, lon_c, 0.5)
    n_grid = np.concatenate([np.arange(-1.0, -0.19, 0.02), np.arange(0.2, 1.001, 0.02)])
    for lon0 in lon_c + np.arange(-40.0, 40.001, 1.0):
        for n in n_grid:
            s = _lcc_grid_misfit(lon2d, lat2d, lon0, n)
            if s < best[0]:
                best = (s, float(lon0), float(n))
    for dl, dn in ((0.1, 0.002), (0.01, 0.0002), (0.001, 0.00002)):
        _, l0, n0 = best
        for lon0 in l0 + np.arange(-10, 10.001) * dl:
            for n in n0 + np.arange(-10, 10.001) * dn:
                if not 0.05 <= abs(n) <= 1.0:
                    continue
                s = _lcc_grid_misfit(lon2d, lat2d, lon0, n)
                if s < best[0]:
                    best = (s, float(lon0), float(n))
    misfit, lon0, n = best
    if not np.isfinite(misfit) or misfit > 1e-3:
        return None
    # 等間隔の検査: 列平均 x を列番号 (間引き後の実インデックス) に直線回帰した残差
    x, y = _lcc_normalized_xy(lon2d, lat2d, lon0, n)
    xc, yc = x.mean(axis=0), y.mean(axis=1)
    for vals, idx in ((xc, jj), (yc, ii)):
        slope, intercept = np.polyfit(idx, vals, 1)
        if slope == 0.0 or np.std(vals - (slope * idx + intercept)) / abs(slope) > 0.02:
            return None
    clon = round(lon0, 2)
    clon = float(round(clon)) if abs(clon - round(clon)) < 0.01 else clon
    clat = round(float((lat2d.min() + lat2d.max()) / 2), 1)
    if abs(n) >= 0.98:
        return {"name": "NorthPolarStereo" if n > 0 else "SouthPolarStereo",
                "central_longitude": clon, "central_latitude": clat,
                "cone_constant": n, "misfit": misfit}
    sign = 1.0 if n > 0 else -1.0
    sp = None
    for p1, p2 in _KNOWN_STANDARD_PARALLELS:
        if abs(_cone_constant(sign * p1, sign * p2) - n) < 0.003:
            sp = [sign * p1, sign * p2]
            break
    if sp is None:
        phi = round(float(np.rad2deg(np.arcsin(n))), 1)
        sp = [phi, phi]
    return {"name": "LambertConformal", "central_longitude": clon,
            "central_latitude": clat, "standard_parallels": sp,
            "cone_constant": n, "misfit": misfit}


def variable_summary(ds: xr.Dataset) -> list[dict[str, str]]:
    """GUIの一覧表示用に、各変数の次元・形状・属性をまとめる。"""
    rows = []
    for name, var in ds.data_vars.items():
        rows.append({
            "変数名": str(name),
            "次元": " × ".join(str(d) for d in var.dims),
            "形状": str(tuple(var.shape)),
            "単位": str(var.attrs.get("units", "")),
            "long_name": str(var.attrs.get("long_name", "")),
        })
    return rows


def coord_summary(ds: xr.Dataset) -> list[dict[str, str]]:
    """GUIの一覧表示用に、各**座標変数**の次元長・範囲・属性をまとめる。

    variable_summary (データ変数用) と対。座標変数を持たない次元 (bare dims) は
    ここには現れない (bare_dims() で別途取得する)。
    """
    rows = []
    for name, coord in ds.coords.items():
        vals = coord.values
        if vals.size == 0:
            rng = ""
        elif vals.size == 1:
            rng = str(vals.reshape(-1)[0])
        else:
            flat = vals.reshape(-1)
            rng = f"{flat[0]} .. {flat[-1]}"
        rows.append({
            "座標名": str(name),
            "次元": " × ".join(str(d) for d in coord.dims),
            "長さ": str(coord.size),
            "範囲": rng,
            "単位": str(coord.attrs.get("units", "")),
            "long_name": str(coord.attrs.get("long_name", "")),
        })
    return rows


def bare_dims(ds: xr.Dataset) -> list[str]:
    """座標変数を持たない次元 (bare dims) の名前一覧を返す。"""
    return [str(d) for d in ds.dims if d not in ds.coords]


def horizontal_map_variables(ds: xr.Dataset, roles: dict[str, str | None] | None = None) -> list[str]:
    """緯度・経度の両次元を持ち、水平断面図を描ける変数の一覧を返す。"""
    roles = roles or detect_coord_roles(ds)
    hdims = horizontal_dims(ds, roles)
    if hdims is None:
        return []
    # 1 次元格子では hdims = (lat 座標名, lon 座標名) なので従来の判定と同じ。
    # curvilinear 格子では lat/lon 座標の dims (y, x) を両方持つ変数
    return [str(name) for name, var in ds.data_vars.items()
            if all(d in var.dims for d in hdims)]


def map_scatter_variables(ds: xr.Dataset, roles: dict[str, str | None] | None = None) -> list[str]:
    """地図散布図 (map_scatter) を描ける変数の一覧を返す。

    lon/lat が変数の**次元**でも (格子データ) **補助座標**でも (地点データ) 可。
    どちらも `lat_name in var.coords and lon_name in var.coords` で拾える。
    """
    roles = roles or detect_coord_roles(ds)
    lat, lon = roles["lat"], roles["lon"]
    if lat is None or lon is None:
        return []
    return [str(name) for name, var in ds.data_vars.items()
            if lat in var.coords and lon in var.coords]


def nongeo_variables(ds: xr.Dataset,
                     roles: dict[str, str | None] | None = None) -> list[str]:
    """緯度・経度の**次元**を持たない数値変数の一覧 (集計系モード用)。

    1次元/2次元プロット(集計) は地図データ非対応 (ユーザー方針 2026-07-07)。
    判定は変数単位: 同じファイル内でも lat/lon 次元を持たない変数 (時系列や
    アンサンブル軸のみの変数) は対象になる。lat/lon が**補助座標**の地点データは
    次元ではないので対象に含まれる。
    """
    roles = roles or detect_coord_roles(ds)
    geo = {roles.get("lat"), roles.get("lon")} - {None}
    if is_curvilinear(ds, roles):
        # lat/lon は dim ではないので、水平面を張る dim (y, x) を持つ変数を除く
        geo = set(horizontal_dims(ds, roles))
    return [str(name) for name, v in ds.data_vars.items()
            if v.ndim >= 1 and v.dtype.kind in "fiu"
            and not (set(v.dims) & geo)]


def track_lonlat_candidates(ds: xr.Dataset) -> tuple[list[str], list[str]]:
    """トラック (軌跡) 用の経度・緯度変数の候補を返す (lon 候補, lat 候補)。

    units 属性 (degrees_east / degrees_north) を優先し、units が無い変数は
    名前 (lon / lat を含む) で拾う。IBTrACS のように lon/lat が座標ではなく
    普通の変数のデータを想定し、data_vars と coords の両方から
    1〜2 次元の変数を対象にする。
    """
    lons: list[str] = []
    lats: list[str] = []
    for name, var in ds.variables.items():
        if var.ndim < 1 or var.ndim > 2:
            continue
        units = str(var.attrs.get("units", "")).lower()
        lname = str(name).lower()
        if units.startswith(("degrees_e", "degree_e")) or \
                (not units and "lon" in lname):
            lons.append(str(name))
        elif units.startswith(("degrees_n", "degree_n")) or \
                (not units and "lat" in lname):
            lats.append(str(name))
    return lons, lats


def track_capable(ds: xr.Dataset) -> bool:
    """トラックレイヤーを描けるか (経度・緯度の変数候補が両方あるか) を返す。

    水平断面図モードの入場ガードで使う (格子変数・map_scatter 変数が無い
    ベストトラック専用データでも地図モードを開けるようにする)。
    """
    lons, lats = track_lonlat_candidates(ds)
    return bool(lons and lats)


def section_variables(ds: xr.Dataset, x_dim: str, y_dim: str) -> list[str]:
    """指定した2次元 (断面の x軸・y軸) を両方持つ変数の一覧を返す。"""
    return [str(name) for name, var in ds.data_vars.items()
            if x_dim in var.dims and y_dim in var.dims]


def _coord_values_match(a, b, *, rtol: float = 1e-7, atol: float = 0.0) -> bool:
    """2つの座標配列が値として一致するかを判定。

    datetime64 と整数は完全一致、浮動小数は allclose で許容する。
    """
    if a.shape != b.shape:
        return False
    try:
        if (np.issubdtype(a.dtype, np.datetime64)
                or np.issubdtype(a.dtype, np.integer)
                or a.dtype.kind in ("U", "S", "O")):
            return bool(np.array_equal(a, b))
        return bool(np.allclose(a, b, rtol=rtol, atol=atol))
    except (TypeError, ValueError):
        return False


_STANDARD_ROLES = ("lat", "lon", "vertical", "time")


def align_dim_names(datasets: dict[str, xr.Dataset], used_ids: list[str],
                     axis_roles: list[str], *,
                     rtol: float = 1e-7, atol: float = 0.0
                     ) -> tuple[dict[str, xr.Dataset], dict[str, dict[str, str]],
                                list[str], list[str]]:
    """used_ids 間で座標 dim 名を canonical (used_ids[0]) に揃え、軸座標の値が一致するか検査。

    処理は2段階:
      1. **dim 名の整列**: 標準役割 (lat/lon/vertical/time) について、各 dataset の
         dim 名を `detect_coord_roles()` で role から逆引きし、canonical と違う名前なら
         **その dataset を rename** する。値の比較は行わない (semantic な
         エイリアスとみなす)。これにより `.sel(lat=...)` などが silent にスキップ
         されるのを防ぐ
      2. **軸座標の値チェック**: `axis_roles` に含まれる役割について、値配列が
         canonical と一致するか確認 (datetime64/整数は equal、float は allclose)。
         - 役割の**座標そのものが無い** → error (その dataset を描けないので描画ブロック)
         - 値が**一致しない** (= 解像度・範囲が異なる) → **info** (エラーにしない)。
           各レイヤーは各自の格子で独立に描かれるため内挿なしに重ねられる

    Args:
        datasets: 全 loaded データセットの dict
        used_ids: 実際に使われている dataset_id の **順序付き** リスト (先頭が canonical)
        axis_roles: 描画軸の役割。値も完全一致が必要 (例 ["lat", "lon"])
        rtol / atol: 浮動小数比較の許容誤差

    Returns:
        renamed_datasets: rename を適用した新しい datasets dict (元は変更しない)
        rename_map: `{dsid: {old_name: new_name}}` rename が起きた dataset のみ
        info_messages: 「`ds1` の `latitude` を `lat` と同一次元とみなします」や
            「格子が異なります … 各レイヤーを各自の解像度で重ね描きします」等 (非ブロック)
        error_messages: 軸役割の座標欠落など、描画をブロックすべきエラー
    """
    if len(used_ids) <= 1:
        return dict(datasets), {}, [], []

    canonical_id = used_ids[0]
    canonical_ds = datasets[canonical_id]
    canonical_roles = detect_coord_roles(canonical_ds)

    out = dict(datasets)
    rename_map: dict[str, dict[str, str]] = {}
    info: list[str] = []
    errors: list[str] = []

    for dsid in used_ids[1:]:
        ds = datasets[dsid]
        ds_roles = detect_coord_roles(ds)

        # Phase 1: 標準役割すべてで dim 名整列 (rename based on role match)
        local_rename: dict[str, str] = {}
        for role in _STANDARD_ROLES:
            canon_dim = canonical_roles.get(role)
            other_dim = ds_roles.get(role)
            if canon_dim and other_dim and other_dim != canon_dim:
                local_rename[other_dim] = canon_dim
        if local_rename:
            ds = ds.rename(local_rename)
            out[dsid] = ds
            rename_map[dsid] = local_rename
            for old, new in local_rename.items():
                info.append(
                    f"`{dsid}` の `{old}` を `{new}` と同一次元とみなします (同じ役割)")
            ds_roles = detect_coord_roles(ds)  # rename 後の roles を取り直す

        # Phase 2: axis_roles の値チェック (rename 適用済みの ds に対して)
        # - 座標役割の**欠落** → errors (その dataset を描画できないので描画ブロック)
        # - 座標値の**不一致** (= 解像度・範囲が異なる) → info (エラーにしない)。
        #   各レイヤーは各自の格子で独立に描かれる (render.py の _draw_* は
        #   layer["dataset_id"] のデータを自分の座標で transform=PlateCarree する)
        #   ため、内挿なしにそのまま重ねられる。描画はブロックしない。
        for role in axis_roles:
            canon_dim = canonical_roles.get(role)
            other_dim = ds_roles.get(role)
            if canon_dim is None:
                continue
            if other_dim is None:
                errors.append(
                    f"`{dsid}` に `{role}` 役割の座標がありません "
                    f"(`{canonical_id}` では `{canon_dim}`)")
                continue
            canon_vals = canonical_ds[canon_dim].values
            other_vals = ds[other_dim].values
            if not _coord_values_match(canon_vals, other_vals,
                                        rtol=rtol, atol=atol):
                info.append(
                    f"`{canonical_id}.{canon_dim}` と `{dsid}.{other_dim}` は"
                    f"格子が異なります ({role} 役割) — 各レイヤーを各自の解像度で"
                    f"重ね描きします")

    return out, rename_map, info, errors
