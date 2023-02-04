import numpy as np
import pandas as pd


class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(
        self,
        n_splits=3,
        train_period_length=126,
        test_period_length=21,
        lookahead=0,
        date_idx="date",
        shuffle=False,
    ):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx

    def split(self, X):
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append(
                [train_start_idx, train_end_idx, test_start_idx, test_end_idx]
            )

        dates = X.reset_index()[[self.date_idx]]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[
                (dates[self.date_idx] > days[train_start])
                & (dates[self.date_idx] <= days[train_end])
            ].index
            test_idx = dates[
                (dates[self.date_idx] > days[test_start])
                & (dates[self.date_idx] <= days[test_end])
            ].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx.to_numpy(), test_idx.to_numpy()

    def get_n_splits(self):
        return self.n_splits


def cache(func):
    def wrapped_func(*args, **kwargs):
        table_name = func.__name__.split("_")[-1]
        parquet_path = f"./data/{table_name}.parquet"
        try:
            table = pd.read_parquet(parquet_path)
        except FileNotFoundError:
            table = func(*args, **kwargs)
            table.to_parquet(parquet_path, index=False)
        return table

    return wrapped_func


def query(db, sql_stmt, **kwargs):
    """
    Query WRDS database, with annoying loading library info suppressed.

    References:
    https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """
    data = db.raw_sql(sql_stmt, date_cols=["date"], params=kwargs)
    return data


def get_crsp(db, permnos):
    """
    Get price data from CRSP.

    Args:
        permnos: an iterative of permnos
    """
    sql_crsp = """
    SELECT
        date,
        permno,
        openprc AS open,
        askhi AS high,
        bidlo AS low,
        prc AS close,
        vol AS volume,
        ret,
        shrout
    FROM
        crsp.dsf
    WHERE
        permno IN %(permnos)s
        AND date >= '1994-01-01'
        AND date <= '2019-12-31'
    ORDER BY
        date, permno;
    """
    # Type check and transform permnos to tuple of str
    if isinstance(permnos, pd.DataFrame):
        permnos = permnos.permno
    if isinstance(permnos, pd.Series):
        permnos = permnos.astype("int").astype("str").pipe(tuple)
    elif any(not isinstance(permno, str) for permno in permnos):
        permnos = tuple(str(permno) for permno in permnos)
    elif not isinstance(permnos, tuple):
        permnos = tuple(permnos)

    # Query data
    crsp = query(db, sql_crsp, permnos=permnos)
    # Fill missing close prices of permno 80539
    crsp.loc[crsp.permno == 80539, "close"] = crsp.loc[
        crsp.permno == 80539, "close"
    ].fillna(method="ffill")
    # Fill other missing values
    crsp = crsp.fillna(
        {
            "open": crsp.close,
            "high": crsp.close,
            "low": crsp.close,
            "volume": 0,
            "ret": 0,
        }
    )
    # Calculate market capitalization
    crsp["cap"] = crsp.close * crsp.shrout
    # Shift market capitalization to avoid look ahead bias
    crsp["cap"] = crsp.groupby("permno").cap.shift(1)
    # Calculate market capiticalization weight
    crsp["w_cap"] = crsp.groupby("date").cap.apply(lambda x: x / x.sum())
    # Convert certain data types to int64
    crsp = crsp.astype({"permno": "int", "shrout": "int", "volume": "int"})
    return crsp


def get_optionmetrics(db, permnos):
    import concurrent.futures

    # Type check and transform permnos to tuple of str
    if isinstance(permnos, pd.DataFrame):
        permnos = permnos.permno
    if isinstance(permnos, pd.Series):
        permnos = permnos.astype("int").astype("str").pipe(tuple)
    elif any(not isinstance(permno, str) for permno in permnos):
        permnos = tuple(str(permno) for permno in permnos)
    elif not isinstance(permnos, tuple):
        permnos = tuple(permnos)

    sql_option_metrics = """
    SELECT
        date,
        permno,
        crsp.secid,
        iv,
        skew_1,
        skew_2
    FROM (
        SELECT
            *
        FROM
            wrdsapps.opcrsphist
        WHERE
            permno IN %(permnos)s) AS crsp
        JOIN (
        SELECT
            a.date,
            a.secid,
            c.iv,
            a.iv - b.iv AS skew_1,
            (g.iv - v.iv) / a.iv AS skew_2
        FROM (
            SELECT
                date,
                secid,
                impl_volatility AS iv
            FROM
                optionm.vsurfd%(year)s
            WHERE
                days = 30
                AND delta = 50) AS a,
            (
            SELECT
                date, secid, impl_volatility AS iv
            FROM
                optionm.vsurfd%(year)s
            WHERE
                days = 30
                AND delta = - 10) AS b,
            (
            SELECT
                date, secid, AVG(impl_volatility) AS iv
            FROM
                optionm.vsurfd%(year)s
            WHERE
                days = 30
                AND abs(delta) <= 50
            GROUP BY
                date,
                secid) AS c,
            (
            SELECT
                date,
                secid,
                impl_volatility AS iv
            FROM
                optionm.vsurfd%(year)s
            WHERE
                days = 30
                AND delta = - 25) AS g,
            (
            SELECT
                date, secid, impl_volatility AS iv
            FROM
                optionm.vsurfd%(year)s
            WHERE
                days = 30
                AND delta = 25) AS v
            WHERE
                a.date = b.date
                AND a.secid = b.secid
                AND a.date = c.date
                AND a.secid = c.secid
                AND a.date = g.date
                AND a.secid = g.secid
                AND a.date = v.date
                AND a.secid = v.secid) AS om ON crsp.secid = om.secid;
    """

    optionmetrics = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_year = [
            executor.submit(
                lambda year: query(db, sql_option_metrics, year=year, permnos=permnos),
                year,
            )
            for year in range(2000, 2020)
        ]
        for future in concurrent.futures.as_completed(future_to_year):
            optionmetrics.append(future.result())
    # Convert certain data types to int64
    optionmetrics = (
        pd.concat(optionmetrics)
        .sort_values(["date", "permno"])
        .astype({"permno": "int", "secid": "int"})
    )
    return optionmetrics


def get_mfis(permnos):
    # Type check and transform permnos to tuple of int
    if isinstance(permnos, pd.DataFrame):
        permnos = permnos.permno
    if isinstance(permnos, pd.Series):
        permnos = permnos.astype("int").pipe(tuple)
    elif any(not isinstance(permno, int) for permno in permnos):
        permnos = tuple(str(permno) for permno in permnos)
    elif not isinstance(permnos, tuple):
        permnos = tuple(permnos)
    # Download MFIS
    mfis = pd.read_csv(
        "https://files.osf.io/v1/resources/btvdh/providers/dropbox/1996-2019/MFIS_1996_2019.csv",
        names=[
            "date",
            "permno",
            "mfis_30",
            "mfis_91",
            "mfis_182",
            "mfis_273",
            "mfis_365",
        ],
        header=0,
        parse_dates=["date"],
        dtype={"permno": "int"},
    )
    mfis = mfis[mfis.permno.isin(permnos)]
    return mfis


def get_glb(permnos):
    # Type check and transform permnos to tuple of int
    if isinstance(permnos, pd.DataFrame):
        permnos = permnos.permno
    if isinstance(permnos, pd.Series):
        permnos = permnos.astype("int").pipe(tuple)
    elif any(not isinstance(permno, int) for permno in permnos):
        permnos = tuple(str(permno) for permno in permnos)
    elif not isinstance(permnos, tuple):
        permnos = tuple(permnos)
    # Download GLB
    glb = pd.read_csv(
        "https://files.osf.io/v1/resources/7xcqw/providers/dropbox/1996-2019/glb30_daily.csv",
        names=["permno", "date", "glb2_30", "glb3_30"],
        header=0,
        parse_dates=["date"],
        dtype={"permno": "int"},
    )
    glb = glb[glb.permno.isin(permnos)]
    return glb


def get_famafrench():
    """
    Get 3-factor, momentum, industry portfolio return from Ken French data library.
    """
    import pandas_datareader as web

    # Transfrom from percentage to nominal value
    factor = (
        web.DataReader(
            "F-F_Research_Data_Factors_daily",
            "famafrench",
            start="1994-01-01",
            end="2019-12-31",
        )[0]
        / 100
    )
    momentum = (
        web.DataReader(
            "F-F_Momentum_Factor_daily",
            "famafrench",
            start="1994-01-01",
            end="2019-12-31",
        )[0]
        / 100
    )
    industry = (
        web.DataReader(
            "17_Industry_Portfolios_daily",
            "famafrench",
            start="1994-01-01",
            end="2019-12-31",
        )[0]
        / 100
    )
    # Substract the risk free rate
    momentum = momentum.sub(factor.RF, axis=0)
    industry = industry.sub(factor.RF, axis=0)
    # Merge into one dataframe
    fama_french = pd.concat([factor, momentum, industry], axis=1)
    fama_french = fama_french.rename(
        columns=lambda x: x.lower().strip().replace("-", "")
    ).rename_axis(index=str.lower)
    return fama_french


def combine_factor(freq, **kwargs):
    factor = pd.DataFrame(
        {name: factor.resample(freq).last().stack() for name, factor in kwargs.items()}
    )
    return factor


def calc_factor(crsp, famafrench, mfis, glb, freq):
    from statsmodels.regression.rolling import RollingOLS

    # Estimate factor exposure
    portfolio = crsp.loc[crsp.date >= "1994-01-01", :].pivot(
        index="date", columns="permno", values="ret"
    )
    portfolio = portfolio.sub(famafrench.loc[portfolio.index, "rf"], axis=0)
    factor = famafrench.loc[portfolio.index, ["mktrf", "smb", "hml", "mom"]].assign(
        alpha=1
    )
    betas = []
    for permno in portfolio:
        ret = portfolio[permno]
        res = RollingOLS(endog=ret, exog=factor, window=756).fit(params_only=True)
        params = res.params
        params["resid"] = ret - factor.mul(params).sum(1)
        params["permno"] = permno
        betas.append(params)
    beta = pd.concat(betas).reset_index().dropna()

    # Pivot data for calculation
    ret = crsp.pivot(index="date", columns="permno", values="ret").sub(
        famafrench.rf, axis=0
    )
    logret = np.log(ret + 1)
    shrout = crsp.pivot(index="date", columns="permno", values="shrout")
    vol = crsp.pivot(index="date", columns="permno", values="volume")
    close = crsp.pivot(index="date", columns="permno", values="close")
    dv = vol.mul(close)
    mktrf = beta.pivot(index="date", columns="permno", values="mktrf")
    resid = beta.pivot(index="date", columns="permno", values="resid")

    # Estimate price trend
    # 1-month cumulative return
    mom_1m = logret.rolling(21).sum()
    # 11-month cumulative returns ending 1-month before
    mom_12m = logret.shift(21).rolling(11 * 21).sum()
    # Cumulative return from months t-6 to t-1 minus months t-12 to t-7.
    mom_6m = logret.shift(21).rolling(5 * 21).sum()
    mom_12m_6m = logret.shift(6 * 21).rolling(5 * 21).sum()
    chmom = mom_6m - mom_12m_6m
    # Max daily returns from calendar month t-1
    maxret = logret.rolling(21).max()
    # Cumulative returns months t-36 to t-13
    mom_36m = logret.shift(12 * 21).rolling(24 * 21).sum()

    # Estimate liquidity
    # Average monthly trading volume for most recent three months divided by number of shares
    turn = vol.shift(21).mean().div(shrout)
    # Monthly std dev of daily share turnover
    turn_std = vol.div(shrout).rolling(21).std()
    # Natural log of market cap
    logcap = np.log(close) + np.log(shrout)
    # Natural log of trading volume times price per share from month t-2
    dolvol = np.log(dv.shift(21).rolling(21).mean())
    # Average of daily (absolute return / dollar volume)
    ill = ret.abs().div(dv)

    # Estimate risk
    # Standard dev of daily returns from month t-1
    retvol = ret.rolling(21).std()
    # Market beta squared
    mktrf_sq = mktrf.pow(2)
    # Idiosyncratic return volatility
    idovol = resid.rolling(756).std()

    # Combine factors to the required frequency
    factor = combine_factor(
        freq,
        mom_1m=mom_1m,
        mom_12m=mom_12m,
        chmom=chmom,
        maxret=maxret,
        mom_36m=mom_36m,
        turn=turn,
        turn_std=turn_std,
        logcap=logcap,
        dolvol=dolvol,
        ill=ill,
        retvol=retvol,
        mktrf_sq=mktrf_sq,
        idovol=idovol,
    )
    if freq == "D":
        beta_ = beta.set_index(["date", "permno"])
        mfis_ = mfis.set_index(["date", "permno"])
        glb_ = glb.set_index(["date", "permno"])
    else:
        beta_ = beta.groupby([pd.Grouper(key="date", freq=freq), "permno"]).mean()
        mfis_ = mfis.groupby([pd.Grouper(key="date", freq=freq), "permno"]).mean()
        glb_ = glb.groupby([pd.Grouper(key="date", freq=freq), "permno"]).mean()
    factor = factor.join(beta_).join(mfis_).join(glb_)

    # Fill missing value with cross sectional mean
    factor = factor.groupby("date").transform(lambda x: x.fillna(x.mean()))
    return factor


def calc_return(crsp, famafrench, freq):
    if freq == "D":
        return crsp.set_index(["date", "permno"]).ret
    else:
        ret = crsp.pivot(index="date", columns="permno", values="ret").sub(
            famafrench.rf, axis=0
        )
        logret = np.log(ret + 1)
        return np.exp(logret.resample(freq).sum().stack()) - 1
