# business_context.py

def load_business_context():
    return {
        # Commodity-level labels
        "PW": "Power",
        "NG": "Natural Gas",
        "CO2": "Carbon Emissions",
        "ELCE": "Electricity",

        # Power-specific business fields
        "BOOK_ATTR6": {
            "B2B": "Business-to-business (commercial customers)",
            "B2C": "Retail customers",
            "ELCE": "Electricity (Renewable generation)",
            "GEN": "Non-renewable Generation",
            "OPT": "Optimization Contracts"
        },
        "BOOK_ATTR7": {
            "ALLOC": "Allocated Cost Portfolio",
            "UNALLOC": "Unallocated Cost Portfolio"
        },
        "BOOK_ATTR8": {
            "NUCLEAR": "Nuclear Generation",
            "HYDRO": "Hydroelectric Generation",
            "SOLAR": "Solar Generation",
            "WIND": "Wind Generation",
            "OTHER": "Waste-based Generation"
        },

        # Primary trading strategy and segment labels
        "TGROUP1": {
            "BACK TO BACK": "Matched buy-sell hedging",
            "INDEX-DA": "Day-ahead index-based trading",
            "FIXED": "Fixed price contract",
            "NUKEFLEET": "Nuclear fleet strategy",
            "GAS ENGINE": "Fast-response flexible gas generation",
            "CHOICE": "Customer choice product",
            "EASYFIX": "Simple fixed-price strategy",
            "BATTERY STORAGE": "Storage-based asset strategy"
        },

        # Secondary trading classification (less used)
        "TGROUP2": {
            "DEFAULT": "Default sub-strategy"
        },

        # Method field (position calc logic)
        "METHOD": {
            "Volumetric": "Physical volume tracking",
            "Financial": "Value-based exposure tracking"
        },

        # Segment label mappings
        "SEGMENT": {
            "ELCE": "Electricity segment",
            "NUCLEAR": "Nuclear generation segment",
            "B2B": "Business customer segment",
            "B2C": "Retail customer segment",
            "OPT": "Options/Derivatives"
        },

        # Book-related labels
        "BOOK": "Trading book",
        "BOOK_ATTR": {
            "BOOK_ATTR6": "Business Classification",
            "BOOK_ATTR7": "Cost Allocation Type",
            "BOOK_ATTR8": "Business Classification"  # Updated as requested
        },

        # Time dimensions
        "BUCKET": {
            "DAH": "Day Ahead",
            "MONTH": "Monthly",
            "QUARTER": "Quarterly",
            "SEASON": "Seasonal"
        },
        "HORIZON": {
            "DAH": "Day Ahead",
            "D+2_EOP": "End of Prompt",
            "EOP_M+1": "Prompt to Next Month",
            "JAN-25": "January 2025",
            "Q1-2025": "Quater 1 2025",
            "SUMMER-25": "Summer 2025",
            "WINTER-28": "Winter 2028"
        },

        # Market access methods (mainly for PW/NG)
        "USR_VAL4": {
            "FIXED": "Fixed price market",
            "N2EX": "UK electricity market",
            "NORDPOOL": "Nordic power market",
            "EPEX": "European power exchange",
            "EDEM": "European day-ahead/season auction market"
        },

        # Volume Fields
        "VOLUME_FIELDS": {
            "VOLUME_BL": "Volume Baseload",
            "VOLUME_PK": "Peak load volume",
            "VOLUME_OFPK": "Off-peak volume",
            "VOLUME_TOTAL": "Total traded volume (NG)",
            "QTY_PHY": "Physical quantity",
            "QTY_FIN": "Financial quantity"
        },

        # Market Value Fields
        "MKTVAL_FIELDS": {
            "MKT_VAL_BL": "Market Value Baseload",
            "MKT_VAL_PK": "Peak load market value",
            "MKT_VAL_OFPK": "Off-peak market value",
            "MKTVAL": "Market value (CO2/NG)"
        },

        # Trade Value Fields
        "TRDVAL_FIELDS": {
            "TRD_VAL_BL": "Base load trade value",
            "TRD_VAL_PK": "Peak load trade value",
            "TRD_VAL_OFPK": "Off-peak trade value",
            "TRDVAL": "Trade value (CO2/NG)"
        },

        # CO2-specific note (can be expanded)
        "CO2_NOTES": {
            "TRDPRC": "Trade price per tonne CO2",
            "VOLUME": "Total emission volume in tonnes"
        },

        # Common interpretation
        "COMMON": {
            "REPORT_DATE": "Report generation date (As-of)",
            "START_DATE": "Start of trade delivery period",
            "END_DATE": "End of trade delivery period"
        },

        # Updated contextual labels
        "LABEL_OVERRIDES": {
            "BOOK_ATTR8": "Business Classification",
            "USR_VAL4": "Route to Market",
            "VOLUME_BL": "Volume Baseload",
            "MKT_VAL_BL": "Market Value Baseload",
            "TGROUP1": "Primary Strategy"
        }
    }