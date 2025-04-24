import streamlit as st
import pandas as pd
import openai
from collections import defaultdict
from datetime import datetime
from openai import OpenAI
from DbUtils.DbOperations import load_business_context


def show_nop_cards(processed_data):
    daily_summary = processed_data.get("daily_nop_summary", [])
    if not daily_summary or len(daily_summary) < 2:
        st.warning("Not enough data to display NOP summary.")
        return

    latest = daily_summary[-1]
    previous = daily_summary[-2]

    latest_date = latest["date"]
    latest_nop = latest["total_volume_bl"]
    delta_nop = latest["delta_volume_bl"]

    latest_mkt_nop = latest["total_market_bl"]
    delta_mkt_nop = latest["delta_market_bl"]


    col1, col2= st.columns(2)
    with col1:
            render_vol_nop_card(latest_nop, delta_nop, latest_date)
    with col2:
            render_mkt_nop_card(latest_mkt_nop, delta_mkt_nop, latest_date)

def render_vol_nop_card(volume, delta, latest_date):
    delta_sign = "+" if delta > 0 else "-" if delta < 0 else ""
    delta_color = "#7dd956" if delta > 0 else "red" if delta < 0 else "gray"

    # HTML template for card
    card_html = f"""
    <div style="background-color: white; padding: 20px; border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05); width: auto;">
        <div style="display: flex; justify-content: space-between; gap: 20px;">
            <div style="flex: 1;">
                <div style="font-size: 16px; color: #333; font-weight: 500;">
                    NOP Volume &#x25B3;
                </div>
                <div style="font-size: 25px; font-weight: 700;">
                    {volume:,.0f}MW
                </div>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 14px; color: #555;">Delta</div>
                <div style="font-size: 25px; font-weight: 700; color: { delta_color };">
                    {delta_sign}{abs(delta):,.0f}MW
                </div>
            </div>
        </div>
        <div style="font-size: 12px; color: #999; margin-top: 10px;">
            As of: {latest_date}
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

def render_mkt_nop_card(volume, delta, latest_date):
    delta_sign = "+" if delta > 0 else "-" if delta < 0 else ""
    delta_color = "#7dd956" if delta > 0 else "red" if delta < 0 else "gray"

    # HTML template for card
    card_html = f"""
    <div style="background-color: white; padding: 20px; border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.05); width: auto;">
        <div style="display: flex; justify-content: space-between; gap: 20px;">
            <div style="flex: 1;">
                <div style="font-size: 16px; color: #333; font-weight: 500;">
                    NOP Market Value &#x25B3;
                </div>
                <div style="font-size: 25px; font-weight: 700;">
                    {volume:,.0f}
                </div>
            </div>
            <div style="flex: 1;">
                <div style="font-size: 14px; color: #555;">Delta</div>
                <div style="font-size: 25px; font-weight: 700; color: { delta_color };">
                    {delta_sign}{abs(delta):,.0f}
                </div>
            </div>
        </div>
        <div style="font-size: 12px; color: #999; margin-top: 10px;">
            As of: {latest_date}
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

def render_summary_card(summary_input, client, conn):
    summary = get_or_generate_summary(summary_input, client, conn)

    card_html = f"""
    <div style="background-color: white; padding: 20px; border-radius: 12px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05); margin-bottom: 10px; width: auto;">
        <div style="font-weight: 600; font-size: 20px; margin-bottom: 10px; color: #333;">
            AI Summary
        </div>
        <div style="font-size: 18px;font-weight: 500; margin-top: 10px;">
            {summary}
        </div>
    </div>
    """

    st.markdown(card_html, unsafe_allow_html=True)

def get_or_generate_summary(summary_input, client ,conn):

    c = conn.cursor()

    today = datetime.today().strftime('%Y-%m-%d')

    c.execute("SELECT summary FROM daily_ai_summary WHERE date = ?", (today,))
    result = c.fetchone()

    if result:
        summary = result[0]
    else:
        summary = generate_segment_summary_from_bl_data(summary_input, client)
        c.execute("INSERT INTO daily_ai_summary (summary, date) VALUES (?, ?)", (summary, today))
        conn.commit()

    conn.close()
    return summary

def generate_segment_summary_from_bl_data(summary, client):
    def get_delta_dict(view_1, view_2, dimension):
        delta_list = []
        for group in view_1:
            for horizon in view_1[group]:
                v1 = view_1[group][horizon].get("total_volume_bl", 0)
                m1 = view_1[group][horizon].get("total_market_value_bl", 0)
                v2 = view_2.get(group, {}).get(horizon, {}).get("total_volume_bl", 0)
                m2 = view_2.get(group, {}).get(horizon, {}).get("total_market_value_bl", 0)
                delta_v = v1 - v2
                delta_m = m1 - m2
                if delta_v != 0 or delta_m != 0:
                    delta_list.append({
                        dimension: group,
                        "Horizon": horizon,
                        "ΔVolume_BL": round(delta_v, 2),
                        "ΔMarket_Value_BL": round(delta_m, 2)
                    })
        return sorted(delta_list, key=lambda x: abs(x["ΔMarket_Value_BL"]), reverse=True)[:3]  # Top 3

    sorted_dates = sorted(summary.keys(), key=lambda d: datetime.strptime(d, "%Y-%m-%d"), reverse=True)
    if len(sorted_dates) < 2:
        return "Not enough data to compare two reporting dates."

    d1, d2 = sorted_dates[:2]
    data1 = summary[d1]
    data2 = summary[d2]

    seg_changes = get_delta_dict(data1["by_segment_and_horizon"], data2["by_segment_and_horizon"], "Segment")
    book_changes = get_delta_dict(data1["by_book_and_horizon"], data2["by_book_and_horizon"], "Book")
    tgroup1_changes = get_delta_dict(data1["by_tgroup1_and_horizon"], data2["by_tgroup1_and_horizon"], "Tgroup1")

    # Format into tabular text
    def format_changes(changes, label):
        header = f"{label} | Horizon | ΔVolume_BL | ΔMkt_Val_BL"
        divider = "-" * len(header)
        rows = "\n".join(
            f"{c.get(label, '')} | {c['Horizon']} | {c['ΔVolume_BL']} | {c['ΔMarket_Value_BL']}" for c in changes
        )
        return f"{header}\n{divider}\n{rows}"

    segment_table = format_changes(seg_changes, "Segment")
    book_table = format_changes(book_changes, "Book")
    tgroup1_table = format_changes(tgroup1_changes, "Tgroup1")

    business_dict = load_business_context()

    business_context_text = "\n".join(
        f"{key}: {value}" if isinstance(value, str)
        else f"{key}: {', '.join(f'{k} = {v}' for k, v in value.items())}"
        for key, value in business_dict.items()
    )

    prompt = f"""
    You are a financial data analyst specializing in commodity trading. Below is structured delta performance data for the latest two reporting days.

    Your task is to:
    1. Analyze changes in baseload volume (VOLUME_BL) and market value (MKT_VAL_BL) between the two report days.
    2. Highlight the top contributing drivers of change, grouped by key business dimensions such as Book, Tgroup1, Segment, or Horizon.
    3. Identify any unusual or unexpected movements worth flagging for management attention.
    4. Always use euro (€) for market value and volume baseload in Megawatt (MW)
    5. Please include values of change in the response

    Strictly use this business glossary to translate technical terms into clear business language:
    {business_context_text}

    Below are changes in Baseload metrics between the two most recent reporting days:

    🔹 Top Movers by Segment:
    {segment_table}

    🔸 Top Movers by Book:
    {book_table}

    🔸 Top Movers by Tgroup1:
    {tgroup1_table}

    Please return a concise, executive-level summary (max 50 words) describing:
    - Key trends in Baseload for volume and market value
    - Which Book, Tgroup1, Segment, or Horizon had the largest impact on changes and show the corresponding values
    - Any anomalies or sharp deviations

    The summary should be suitable for a business dashboard and easily digestible by senior leadership.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a senior energy analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error generating summary: {str(e)}"
