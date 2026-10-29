from typing import Optional
import geopandas as gpd
import numpy as np
import pandas as pd

def apply_maatregelen(
    geodata: gpd.GeoDataFrame,
    maatregelen_gdf: Optional[gpd.GeoDataFrame],
    catchments: gpd.GeoDataFrame,
    log=print,
) -> gpd.GeoDataFrame:
    """
    Pas maatregelen toe op geodata:
    - behoud alleen maatregelen met niet-lege 'type'
    - sla expliciet als 'niet_gemodelleerd' gemarkeerde objecten over
    - behoud enkel maatregelen die een stroomgebied snijden
    - log hoeveel maatregelen verwerkt worden
    - pas DTM / helling / runoff / Manning aan per maatregeltype.
    """
    if maatregelen_gdf is None:
        return geodata

    if "type" not in maatregelen_gdf.columns:
        log("Maatregelen-laag heeft geen kolom 'type' – maatregelen worden overgeslagen.")
        return geodata

    # Enkel maatregelen met type
    maatregelen_gdf = maatregelen_gdf[~maatregelen_gdf["type"].isna()].copy()
    niet_gemodelleerd = maatregelen_gdf["type"].eq("niet_gemodelleerd")
    if niet_gemodelleerd.any():
        log(
            f"Maatregelen: {int(niet_gemodelleerd.sum())} object(en) zijn "
            "nog niet gemodelleerd en worden niet doorgerekend."
        )
        maatregelen_gdf = maatregelen_gdf.loc[~niet_gemodelleerd].copy()
    if maatregelen_gdf.empty:
        log("Geen door te rekenen maatregelen over – niets toe te passen.")
        return geodata

    # Enkel maatregelen die een stroomgebied snijden
    maatregelen_gdf = gpd.sjoin(
        maatregelen_gdf,
        catchments[["uitstroompunt_nummer", "geometry"]],
        how="inner",
        predicate="intersects",
    ).drop(columns=["index_right"])

    if maatregelen_gdf.empty:
        log("Geen maatregelen die stroomgebieden snijden – niets toe te passen.")
        return geodata

    # Log aantal maatregelen
    n_maatregelen = len(maatregelen_gdf)
    log(
        f"Stap: maatregelen toepassen – {n_maatregelen} maatregel(en) met type + intersectie met stroomgebieden."
    )

    # Zorg dat x/y aanwezig zijn
    if "x" not in geodata.columns or "y" not in geodata.columns:
        geodata = geodata.copy()
        geodata["x"] = geodata.geometry.x
        geodata["y"] = geodata.geometry.y

    # DTM-kolom die in het model wordt gebruikt
    if "DTM_5m" not in geodata.columns:
        raise KeyError(
            "Kolom 'DTM_5m' ontbreekt in geodata; "
            "nodig voor DTM-aanpassingen door maatregelen."
        )

    # Minimum DTM per stroomgebied
    min_dtm_per_catchment = geodata.groupby("catchment_id")["DTM_5m"].min()

    # Doorloop alle maatregelen
    for _, maatregel in maatregelen_gdf.iterrows():
        geom = maatregel.geometry
        geom_type = geom.geom_type
        mtype = str(maatregel.type).lower()

        # --------------------------------------------------
        # PUNT-MAATREGELEN
        # --------------------------------------------------
        if geom_type in ("Point", "MultiPoint"):
            if mtype == "drempel":
                # Zoek dichtstbijzijnde cel in geodata (op basis van x,y)
                px, py = geom.x, geom.y

                xs = np.sort(geodata["x"].unique())
                ys = np.sort(geodata["y"].unique())

                col_idx = int(np.abs(xs - px).argmin())
                row_idx = int(np.abs(ys - py).argmin())

                x_val = xs[col_idx]
                y_val = ys[row_idx]

                candidate = geodata[(geodata["x"] == x_val) & (geodata["y"] == y_val)]
                if candidate.empty:
                    continue

                cell_idx = candidate.index[0]
                catchment_id = geodata.loc[cell_idx, "catchment_id"]

                # laagste helling in het stroomgebied
                min_slope = geodata.loc[
                    geodata["catchment_id"] == catchment_id, "D8slope"
                ].min()
                geodata.at[cell_idx, "D8slope"] = min_slope

                min_dtm = min_dtm_per_catchment.loc[catchment_id]
                geodata.at[cell_idx, "DTM_5m"] = min_dtm - 0.3

        # --------------------------------------------------
        # LIJN-MAATREGELEN
        # --------------------------------------------------
        if geom_type in ("LineString", "MultiLineString"):
            buffer = geom.buffer(5)

            if mtype == "gracht":
                idxs = geodata[geodata.geometry.within(buffer)].index
                if len(idxs) > 0:
                    min_slope = geodata.loc[idxs, "D8slope"].min()
                    geodata.loc[idxs, "D8slope"] = min_slope
                    for cell_idx in idxs:
                        catchment_id = geodata.loc[cell_idx, "catchment_id"]
                        min_dtm = min_dtm_per_catchment.loc[catchment_id]
                        geodata.at[cell_idx, "DTM_5m"] = min_dtm - 0.3

            if mtype == "houthaksel":
                idxs = geodata[geodata.geometry.within(buffer)].index
                if len(idxs) > 0:
                    min_slope = geodata.loc[idxs, "D8slope"].min()
                    geodata.loc[idxs, "D8slope"] = min_slope
                    max_manning = geodata.loc[idxs, "manning_n"].max()
                    geodata.loc[idxs, "manning_n"] = max_manning

            if mtype == "houtkant":
                idxs = geodata[geodata.geometry.within(buffer)].index
                if len(idxs) > 0:
                    min_slope = geodata.loc[idxs, "D8slope"].min()
                    geodata.loc[idxs, "D8slope"] = min_slope
                    geodata.loc[idxs, "manning_n"] = geodata.loc[idxs, "manning_n"] * 2

            if mtype == "infiltratiestraat":
                idxs = geodata[geodata.geometry.within(buffer)].index
                if len(idxs) > 0:
                    geodata.loc[idxs, "runoff_ori_clipped"] = 0.0
                    geodata.loc[idxs, "manning_n"] = geodata.loc[idxs, "manning_n"] * 2

            if mtype == "swale":
                idxs = geodata[geodata.geometry.within(buffer)].index
                if len(idxs) > 0:
                    geodata.loc[idxs, "runoff_ori_clipped"] = 0.0
                    for cell_idx in idxs:
                        catchment_id = geodata.loc[cell_idx, "catchment_id"]
                        min_dtm = min_dtm_per_catchment.loc[catchment_id]
                        geodata.at[cell_idx, "DTM_5m"] = min_dtm - 0.2

        # --------------------------------------------------
        # POLYGOON-MAATREGELEN
        # --------------------------------------------------
        if geom_type in ("Polygon", "MultiPolygon"):
            buffer = geom

            if mtype == "bufferbekken":
                idxs = geodata[geodata.geometry.within(buffer)].index
                if len(idxs) > 0:
                    geodata.loc[idxs, "runoff_ori_clipped"] = 0.0

                    if pd.notna(maatregel.get("volume")):
                        totale_oppervlakte = len(idxs) * (5 * 5)  # 5m x 5m
                        volume_m3 = float(maatregel.volume)
                        depth_lowering = volume_m3 / totale_oppervlakte
                        for cell_idx in idxs:
                            catchment_id = geodata.loc[cell_idx, "catchment_id"]
                            min_dtm = min_dtm_per_catchment.loc[catchment_id]
                            geodata.at[cell_idx, "DTM_5m"] = (
                                min_dtm - depth_lowering
                            )
                    else:
                        for cell_idx in idxs:
                            catchment_id = geodata.loc[cell_idx, "catchment_id"]
                            min_dtm = min_dtm_per_catchment.loc[catchment_id]
                            geodata.at[cell_idx, "DTM_5m"] = min_dtm - 0.5

            if mtype == "grasbufferstrook":
                idxs = geodata[geodata.geometry.within(buffer)].index
                if len(idxs) > 0:
                    geodata.loc[idxs, "runoff_ori_clipped"] = 0.0
                    geodata.loc[idxs, "D8slope"] = geodata.loc[idxs, "D8slope"] / 2.0
                    max_manning = geodata.loc[idxs, "manning_n"].max()
                    geodata.loc[idxs, "manning_n"] = max_manning

            if mtype == "ontharding":
                idxs = geodata[geodata.geometry.within(buffer)].index
                for cell_idx in idxs:
                    ro_val = geodata.at[cell_idx, "runoff_ori_clipped"]
                    if ro_val > 50.0:
                        nieuw_ro = 50.0 + (ro_val - 50.0) / 2.0
                        geodata.at[cell_idx, "runoff_ori_clipped"] = nieuw_ro

            if mtype in ("sponsakker", "sponstuin"):
                idxs = geodata[geodata.geometry.within(buffer)].index
                if len(idxs) > 0:
                    geodata.loc[idxs, "runoff_ori_clipped"] = 0.0
                    geodata.loc[idxs, "manning_n"] = geodata.loc[idxs, "manning_n"] * 2

    return geodata
