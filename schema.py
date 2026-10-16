from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Categories learned by the model's encoder during training
CODE_GENDER_VALUES = Literal[
    "F",
    "M",
    "XNA",
]
NAME_EDUCATION_TYPE_VALUES = Literal[
    "Academic degree",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Secondary / secondary special",
]
NAME_FAMILY_STATUS_VALUES = Literal[
    "Civil marriage",
    "Married",
    "Separated",
    "Single / not married",
    "Unknown",
    "Widow",
]
FLAG_OWN_CAR_VALUES = Literal[
    "N",
    "Y",
]
NAME_CONTRACT_TYPE_VALUES = Literal[
    "Cash loans",
    "Revolving loans",
]
OCCUPATION_TYPE_VALUES = Literal[
    "Accountants",
    "Cleaning staff",
    "Cooking staff",
    "Core staff",
    "Drivers",
    "HR staff",
    "High skill tech staff",
    "IT staff",
    "Laborers",
    "Low-skill Laborers",
    "Managers",
    "Medicine staff",
    "Private service staff",
    "Realty agents",
    "Sales staff",
    "Secretaries",
    "Security staff",
    "Unknown",
    "Waiters/barmen staff",
]
ORGANIZATION_TYPE_VALUES = Literal[
    "Advertising",
    "Agriculture",
    "Bank",
    "Business Entity Type 1",
    "Business Entity Type 2",
    "Business Entity Type 3",
    "Cleaning",
    "Construction",
    "Culture",
    "Electricity",
    "Emergency",
    "Government",
    "Hotel",
    "Housing",
    "Industry: type 1",
    "Industry: type 10",
    "Industry: type 11",
    "Industry: type 12",
    "Industry: type 13",
    "Industry: type 2",
    "Industry: type 3",
    "Industry: type 4",
    "Industry: type 5",
    "Industry: type 6",
    "Industry: type 7",
    "Industry: type 8",
    "Industry: type 9",
    "Insurance",
    "Kindergarten",
    "Legal Services",
    "Medicine",
    "Military",
    "Mobile",
    "Other",
    "Police",
    "Postal",
    "Realtor",
    "Religion",
    "Restaurant",
    "School",
    "Security",
    "Security Ministries",
    "Self-employed",
    "Services",
    "Telecom",
    "Trade: type 1",
    "Trade: type 2",
    "Trade: type 3",
    "Trade: type 4",
    "Trade: type 5",
    "Trade: type 6",
    "Trade: type 7",
    "Transport: type 1",
    "Transport: type 2",
    "Transport: type 3",
    "Transport: type 4",
    "University",
    "XNA",
]
NAME_INCOME_TYPE_VALUES = Literal[
    "Businessman",
    "Commercial associate",
    "Maternity leave",
    "Pensioner",
    "State servant",
    "Student",
    "Unemployed",
    "Working",
]


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
    EXT_SOURCE_2: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_1: float | None = Field(default=None, ge=0, le=1)
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
    CODE_GENDER: CODE_GENDER_VALUES | None = None
    NAME_EDUCATION_TYPE: NAME_EDUCATION_TYPE_VALUES | None = None
    NAME_FAMILY_STATUS: NAME_FAMILY_STATUS_VALUES | None = None
    FLAG_OWN_CAR: FLAG_OWN_CAR_VALUES | None = None
    NAME_CONTRACT_TYPE: NAME_CONTRACT_TYPE_VALUES | None = None
    OCCUPATION_TYPE: OCCUPATION_TYPE_VALUES | None = None
    ORGANIZATION_TYPE: ORGANIZATION_TYPE_VALUES | None = None
    NAME_INCOME_TYPE: NAME_INCOME_TYPE_VALUES | None = None