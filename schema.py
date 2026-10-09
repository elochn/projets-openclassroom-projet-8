from pydantic import BaseModel, ConfigDict, Field

# Describes the JSON body expected by POST /predict:
class ClientData(BaseModel):
    # Reject any field that is not one of the 60 model features (typos included)
    model_config = ConfigDict(extra="forbid")

    # --- Required fields ---
    # gt=0 means "greater than 0" when it must be strictly positive
    AMT_CREDIT: float = Field(gt=0)
    AMT_ANNUITY: float = Field(gt=0)
    
    # Age in days before the application, so always negative:
    # ge = "greater or equal", le = "less or equal"
    # between 100 years (-36525 days) and 18 years (-6575 days)
    DAYS_BIRTH: float = Field(ge=-36525, le=-6575)

    # --- Optional numeric fields ---
    # A missing value is imputed by the pipeline.
    # Days of employment before the application: zero or negative
    DAYS_EMPLOYED: float | None = Field(default=None, le=0) # le = "less or equal"
    EXT_SOURCE_2: float | None = None
    EXT_SOURCE_3: float | None = None
    EXT_SOURCE_1: float | None = None
    PAYMENT_RATE: float | None = None
    AMT_GOODS_PRICE: float | None = None
    PREV_DAYS_LATE_MEAN: float | None = None
    PREV_PREV_DECISION_Refused_MEAN: float | None = None
    PREV_CNT_INSTALMENT_FUTURE_MAX: float | None = None
    PREV_CNT_INSTALMENT_FUTURE_MEAN: float | None = None
    BUREAU_AMT_CREDIT_SUM_DEBT_MEAN: float | None = None
    ANNUITY_INCOME_PERC: float | None = None
    OWN_CAR_AGE: float | None = None
    PREV_DAYS_LATE_MIN: float | None = None
    PREV_AMT_ANNUITY_MEAN: float | None = None
    PREV_DAYS_LATE_MAX: float | None = None
    PREV_PAYMENT_DIFF_MEAN: float | None = None
    PREV_POS_STATUS_Active_SUM: float | None = None
    BUREAU_DAYS_CREDIT_MAX: float | None = None
    FLAG_DOCUMENT_3: float | None = None
    DAYS_ID_PUBLISH: float | None = None
    BUREAU_DAYS_CREDIT_ENDDATE_MAX: float | None = None
    PREV_CNT_PAYMENT_MAX: float | None = None
    PREV_DAYS_LATE_SUM: float | None = None
    REGION_RATING_CLIENT_W_CITY: float | None = None
    BUREAU_DAYS_ENDDATE_FACT_MAX: float | None = None
    DAYS_EMPLOYED_PERC: float | None = None
    PREV_AMT_ANNUITY_MIN: float | None = None
    PREV_AMT_DOWN_PAYMENT_SUM: float | None = None
    DEF_60_CNT_SOCIAL_CIRCLE: float | None = None
    PREV_AMT_DOWN_PAYMENT_MAX: float | None = None
    BUREAU_BUREAU_ACTIVE_Closed_MEAN: float | None = None
    BUREAU_DAYS_CREDIT_MEAN: float | None = None
    BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN: float | None = None
    DAYS_LAST_PHONE_CHANGE: float | None = None
    PREV_CNT_PAYMENT_MEAN: float | None = None
    PREV_SK_DPD_DEF_POS_MEAN: float | None = None
    PREV_DAYS_LAST_DUE_1ST_VERSION_MAX: float | None = None
    PREV_PREV_DECISION_Approved_MEAN: float | None = None
    PREV_DAYS_FIRST_DUE_MIN: float | None = None
    INCOME_CREDIT_PERC: float | None = None
    REG_CITY_NOT_LIVE_CITY: float | None = None
    PREV_AMT_BALANCE_MAX: float | None = None
    BUREAU_AMT_CREDIT_SUM_MAX: float | None = None
    PREV_AMT_DRAWINGS_CURRENT_MAX: float | None = None
    BUREAU_BUREAU_ACTIVE_Active_SUM: float | None = None
    BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN: float | None = None
    PREV_AMT_DOWN_PAYMENT_MEAN: float | None = None
    PREV_MONTHS_BALANCE_SUM: float | None = None

    # --- Optional categorical fields (text) ---
    CODE_GENDER: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    FLAG_OWN_CAR: str | None = None
    NAME_CONTRACT_TYPE: str | None = None
    OCCUPATION_TYPE: str | None = None
    ORGANIZATION_TYPE: str | None = None
    NAME_INCOME_TYPE: str | None = None