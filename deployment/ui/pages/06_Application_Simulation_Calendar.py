from curses.ascii import alt
import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime as _dt, timedelta
import calendar as cal
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import requests
# from dotenv import load_dotenv


st.set_page_config(page_title="Calendar", page_icon="🗓️", layout="wide")
st.title("Calendar")
st.caption("Visualize events on a calendar")
# --- Spotify Theme CSS ---
st.markdown("""
<style>
/* Background & general text */
body, .stApp {
    background-color: #121212;
    color: #FFFFFF;
}

/* Calendar header */
.calendar-header {
    background-color: #1DB954;
    color: #121212;
    padding: 10px;
    border-radius: 10px 10px 0 0;
    text-align: center;
    font-weight: bold;
    margin-bottom: 2px;
}

/* Day cell */
.calendar-day {
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 8px;
    min-height: 100px;
    transition: all 0.3s ease;
}

/* Hover effect */
.calendar-day:hover {
    box-shadow: 0 4px 12px rgba(29, 185, 84, 0.5);
    transform: translateY(-2px);
}

/* Today highlight */
.today-highlight {
    background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
    border: 2px solid #1DB954;
    box-shadow: 0 4px 12px rgba(29, 185, 84, 0.4);
}

/* Weekend day */
.weekend-day {
    background-color: #1e1e1e;
}

.event-badge {
    font-size: 0.65em;
    padding: 3px 5px;
    margin: 2px 0;
    border-radius: 4px;
    background-color: rgba(0,0,0,0.3); /* soft dark fallback */
    color: #000000;             /* teks hitam */
    display: block;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)


# Backend API configuration
load_dotenv()  # Load .env file if it exists

BACKEND_URL = os.getenv("BACKEND_API", "http://127.0.0.1:8000")


def calendar_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ui/pages
    ui_dir = os.path.dirname(current_dir)  # .../ui
    return os.path.join(os.path.dirname(ui_dir), "data", "calendar", "events.json")


def load_events():
    path = calendar_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        # migrate old file if exists
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pages_dir = os.path.dirname(current_dir)
        ui_dir = os.path.dirname(pages_dir)
        dev_dir = os.path.dirname(ui_dir)
        old_path = os.path.join(dev_dir, "data", "calendar.json")
        if os.path.exists(old_path):
            with open(old_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as wf:
                json.dump(data, wf, ensure_ascii=False, indent=2)
            return data
    except Exception as e:
        st.error(f"Failed to load calendar: {e}")
    return []


# def save_events(events):
#     # events_example = {
#     #     "title": "Team Standup",
#     #     "start": "2025-11-01T09:00:00",
#     #     "end": "2025-11-01T09:30:00",
#     #     "description": "Daily team sync meeting",
#     #     "location": "Conference Room A",
#     #     "label": "meeting"
#     # }

#     # to do: add elements like urgency_level, meeting_url, etc.
#     path = calendar_path()
#     try:
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(events, f, ensure_ascii=False, indent=2)
#         return True
#     except Exception as e:
#         st.error(f"Failed to save calendar: {e}")
#         return False


def create_event_api(event_data):
    """Call backend API to create a new event"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/create",
            json=event_data,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend server. Please ensure the backend is running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Failed to create event: {str(e)}")
        return None


def get_events_for_date(events, date):
    """Get all events that occur on a specific date"""
    date_events = []
    for event in events:
        try:
            start = pd.to_datetime(event["start"]).date()
            end = pd.to_datetime(event["end"]).date()
            if start <= date <= end:
                date_events.append(event)
        except:
            continue
    return date_events


def render_calendar(events, year, month):
    """Render a monthly calendar view with events"""
    
    # Get calendar for the month
    month_calendar = cal.monthcalendar(year, month)
    month_name = cal.month_name[month]
    
    # Label colors with better aesthetics
    label_colors = {
        "meeting": "🔵",
        "appointment": "🟢", 
        "deadline": "🔴",
        "reminder": "🟡",
        "other": "⚫"
    }
    category_colors = {
        "meeting": "#A3C4F3",      # soft blue
        "appointment": "#B8E6B8",  # soft green
        "deadline": "#F5A3A3",     # soft red
        "reminder": "#FFE5A3",     # soft yellow
        "other": "#D3D3D3",        # soft gray
        "update": "#D8B3F5"        # soft purple
    }

    # Opacity based on urgency
    priority_opacity = {
        "low": 0.2,
        "medium": 0.4,
        "high": 0.6
    } 
    
    st.subheader(f"{month_name} {year}")
    
    # Create header row
    cols = st.columns(7)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, day in enumerate(days):
        with cols[i]:
            is_weekend = i >= 5
            header_color = "#95a5a6" if is_weekend else "#667eea"
            st.markdown(
                f"<div style='background-color: {header_color}; color: white; padding: 8px; "
                f"text-align: center; font-weight: bold; border-radius: 5px;'>{day}</div>",
                unsafe_allow_html=True
            )
    
    # Render calendar grid
    for week in month_calendar:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                if day == 0:
                    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
                else:
                    current_date = _dt(year, month, day).date()
                    day_events = get_events_for_date(events, current_date)
                    
                    # Check if it's today, weekend
                    is_today = current_date == _dt.now().date()
                    is_weekend = i >= 5
                    
                    # Determine background styling
                    if is_today:
                        bg_style = "background: linear-gradient(135deg, #fff4e6 0%, #ffe8cc 100%); border: 2px solid #ff9f43; box-shadow: 0 4px 12px rgba(255, 159, 67, 0.3);"
                        day_style = f"<span style='color: #e67e22; font-weight: bold; font-size: 1.2em;'>{day}</span>"
                    elif is_weekend:
                        bg_style = "background-color: #f8f9fa; border: 1px solid #dee2e6;"
                        day_style = f"<span style='color: #95a5a6;'>{day}</span>"
                    else:
                        bg_style = "background-color: white; border: 1px solid #e0e0e0;"
                        day_style = f"<span style='color: #2d3436;'>{day}</span>"
                    
                    # Create container for the day
                    with st.container():
                        # Day number
                        st.markdown(
                            f"<div style='text-align: center; padding: 8px; {bg_style} "
                            f"border-radius: 8px; min-height: 100px;'>"
                            f"{day_style}",
                            unsafe_allow_html=True
                        )
                        
                        # # Show event indicators
                        # Show event indicators with color based on category & priority
                        if day_events:
                            # Use first event or choose the one with highest urgency
                            event = day_events[0]  # or max(day_events, key=lambda x: x.get("urgency_score",0))
                            category = event.get("label", "other")
                            urgency = event.get("urgency_level", "low")
                            
                            base_color = category_colors.get(category, "#9E9E9E")
                            opacity = priority_opacity.get(urgency, 0.2)
                            
                            # Use RGBA for background color
                            bg_style = f"background-color: {base_color}; opacity: {opacity}; border-radius: 8px; min-height: 100px; padding: 8px;"
                            
                            # Day number
                            day_style = f"<span style='font-weight: bold; font-size: 1.2em;'>{day}</span>"
                            
                            # Create container with background color
                            event_html = f"<div style='{bg_style}'>{day_style}<br>"
                            
                            # Display up to 3 events
                            for ev in day_events[:3]:
                                event_html += f"<div class='event-badge'>{ev.get('title','')}</div>"
                            if len(day_events) > 3:
                                event_html += f"<div class='event-badge'>+{len(day_events)-3} more</div>"
                            
                            event_html += "</div>"
                            
                            st.markdown(event_html, unsafe_allow_html=True)

                        # if day_events:
                        #     event_html = ""
                        #     for event in day_events[:3]:  # Show max 3 events
                        #         label_color = label_colors.get(event.get("label", "other"), "⚫")
                        #         title = event.get("title", "")
                        #         event_html += f"<div class='event-badge' style='color: #7f8c8d'>{label_color} {title}</div>"
                            
                        #     if len(day_events) > 3:
                        #         event_html += f"<div class='event-badge' style='font-style: italic; color: #7f8c8d;'>+{len(day_events) - 3} more</div>"
                            
                        #     st.markdown(event_html + "</div>", unsafe_allow_html=True)
                        # else:
                        #     st.markdown("</div>", unsafe_allow_html=True)


def render_timeline(events, year, month):
    """Render a Gantt-style timeline for events"""
    st.subheader(f"Timeline View")
    
    if not events:
        st.info("No events to display in timeline")
        return
    
    try:
        df = pd.DataFrame(events)
        df["start"] = pd.to_datetime(df["start"], errors="coerce")
        df["end"] = pd.to_datetime(df["end"], errors="coerce")
        
        # Remove rows with invalid dates
        df = df.dropna(subset=["start", "end"])
        
        if df.empty:
            st.info("No valid events to display")
            return
        
        # Filter controls
        col_filter1, col_filter2, col_filter3 = st.columns([2, 2, 1])
        
        with col_filter1:
            # Default to current month range
            start_of_month = _dt(year, month, 1).date()
            if month == 12:
                end_of_month = _dt(year + 1, 1, 1).date() - timedelta(days=1)
            else:
                end_of_month = _dt(year, month + 1, 1).date() - timedelta(days=1)
            
            start_filter = st.date_input(
                "From Date",
                value=start_of_month,
                key="timeline_start"
            )
        
        with col_filter2:
            end_filter = st.date_input(
                "To Date",
                value=end_of_month,
                key="timeline_end"
            )
        
        with col_filter3:
            view_all = st.checkbox("View All", value=False)
        
        # Apply filters
        if not view_all:
            mask = (
                (df["start"].dt.date <= end_filter) & 
                (df["end"].dt.date >= start_filter)
            )
            df_filtered = df[mask].copy()
        else:
            df_filtered = df.copy()
        
        if df_filtered.empty:
            st.warning("No events found in the selected date range")
            return
        
        # Sort by start date
        df_filtered = df_filtered.sort_values("start")
        
        # Color mapping for labels
        color_map = {
            "meeting": "#2196F3",
            "appointment": "#4CAF50",
            "deadline": "#F44336",
            "reminder": "#FFC107",
            "other": "#9E9E9E"
        }
        
        df_filtered["color"] = df_filtered["label"].map(color_map)
        
        # Create Gantt chart using Plotly
        fig = go.Figure()
        
        # Add bars for each event
        for idx, row in df_filtered.iterrows():
            # Calculate duration
            duration = (row["end"] - row["start"]).total_seconds() / 3600  # hours
            
            fig.add_trace(go.Bar(
                x=[duration],
                y=[row["title"]],
                orientation='h',
                base=row["start"],
                marker=dict(color=row["color"]),
                hovertemplate=(
                    f"<b>{row['title']}</b><br>"
                    f"<b>Start:</b> {row['start'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"<b>End:</b> {row['end'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"<b>Duration:</b> {duration:.1f} hours<br>"
                    f"<b>Label:</b> {row['label']}<br>"
                    f"<b>Location:</b> {row.get('location', 'N/A')}<br>"
                    f"<b>Description:</b> {row.get('description', 'N/A')}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
                name=row["title"]
            ))
        
        # Add vertical line for today if in range
        today = _dt.now()
        if (not view_all and start_filter <= today.date() <= end_filter) or view_all:
            fig.add_vline(
                x=today,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text="Today",
                annotation_position="top"
            )
        
        # Update layout
        fig.update_layout(
            title="Event Timeline (Gantt Chart)",
            xaxis_title="Date & Time",
            yaxis_title="Events",
            height=max(400, len(df_filtered) * 50),
            hovermode="closest",
            xaxis=dict(
                type='date',
                tickformat="%b %d, %Y",
            ),
            yaxis=dict(
                categoryorder="array",
                categoryarray=df_filtered["title"].tolist()[::-1]  # Reverse for better readability
            ),
            bargap=0.3,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(l=150, r=20, t=60, b=60)
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend for label colors
        st.markdown("**Legend:**")
        legend_cols = st.columns(5)
        for idx, (label, color) in enumerate(color_map.items()):
            with legend_cols[idx]:
                st.markdown(
                    f"<div style='background-color: {color}; color: white; padding: 5px; "
                    f"border-radius: 5px; text-align: center; font-size: 0.8em;'>"
                    f"{label.capitalize()}</div>",
                    unsafe_allow_html=True
                )
        
        # Event details table
        with st.expander("📋 View Event Details"):
            display_df = df_filtered[{
                "title", "start", "end", "label", "location", "description"
            }].copy()
            
            # Format datetime columns
            display_df["start"] = display_df["start"].dt.strftime("%Y-%m-%d %H:%M")
            display_df["end"] = display_df["end"].dt.strftime("%Y-%m-%d %H:%M")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Statistics
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        with col_stat1:
            st.metric("Total Events", len(df_filtered))
        with col_stat2:
            most_common_label = df_filtered["label"].mode()[0] if not df_filtered.empty else "N/A"
            st.metric("Most Common Type", most_common_label.capitalize())
        with col_stat3:
            avg_duration = df_filtered.apply(
                lambda row: (row["end"] - row["start"]).total_seconds() / 3600, 
                axis=1
            ).mean()
            st.metric("Avg Duration (hrs)", f"{avg_duration:.1f}")
        with col_stat4:
            unique_locations = df_filtered["location"].nunique()
            st.metric("Unique Locations", unique_locations)
        
    except Exception as e:
        st.error(f"Failed to render timeline: {e}")
        st.exception(e)


if "calendar_events" not in st.session_state:
    st.session_state.calendar_events = load_events()

if "calendar_date" not in st.session_state:
    st.session_state.calendar_date = _dt.now()

# Calendar navigation with better UI
st.markdown("### Navigate Calendar")

# Create a more intuitive navigation layout
col_year_prev, col_month_prev, col_current, col_month_next, col_year_next = st.columns([1, 1, 3, 1, 1])

with col_year_prev:
    if st.button("Back 1 Year", help="Previous Year", use_container_width=True):
        current = st.session_state.calendar_date
        st.session_state.calendar_date = current.replace(year=current.year - 1)
        st.rerun()

with col_month_prev:
    if st.button("◀", help="Previous Month", use_container_width=True):
        current = st.session_state.calendar_date
        if current.month == 1:
            st.session_state.calendar_date = current.replace(year=current.year - 1, month=12)
        else:
            st.session_state.calendar_date = current.replace(month=current.month - 1)
        st.rerun()

with col_current:
    # Separate month and year selection for better UX
    col_month_select, col_year_select = st.columns(2)
    
    with col_month_select:
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        selected_month_idx = st.selectbox(
            "Month",
            options=range(12),
            index=st.session_state.calendar_date.month - 1,
            format_func=lambda x: months[x],
            key="month_selector",
            label_visibility="collapsed"
        )
    
    with col_year_select:
        current_year = st.session_state.calendar_date.year
        year_range = list(range(2020, 2031))
        selected_year = st.selectbox(
            "Year",
            options=year_range,
            index=year_range.index(current_year) if current_year in year_range else 5,
            key="year_selector",
            label_visibility="collapsed"
        )
    
    # Update calendar date if selection changed
    if selected_month_idx + 1 != st.session_state.calendar_date.month or selected_year != st.session_state.calendar_date.year:
        st.session_state.calendar_date = st.session_state.calendar_date.replace(
            year=selected_year, 
            month=selected_month_idx + 1
        )
        st.rerun()

with col_month_next:
    if st.button("▶", help="Next Month", use_container_width=True):
        current = st.session_state.calendar_date
        if current.month == 12:
            st.session_state.calendar_date = current.replace(year=current.year + 1, month=1)
        else:
            st.session_state.calendar_date = current.replace(month=current.month + 1)
        st.rerun()

with col_year_next:
    if st.button("Jump 1 Year", help="Next Year", use_container_width=True):
        current = st.session_state.calendar_date
        st.session_state.calendar_date = current.replace(year=current.year + 1)
        st.rerun()

# Quick jump to today button
col_today, col_spacer = st.columns([1, 6])
with col_today:
    if st.button("Today", use_container_width=True):
        st.session_state.calendar_date = _dt.now()
        st.rerun()

st.markdown("---")

# Render calendar and timeline stacked vertically
events = st.session_state.calendar_events

# Calendar view (full width)
render_calendar(events, st.session_state.calendar_date.year, st.session_state.calendar_date.month)

st.markdown("---")

# Event list for selected month
st.subheader("Events This Month")
if events:
    try:
        dfc = pd.DataFrame(events)
        dfc["start"] = pd.to_datetime(dfc["start"], errors="coerce")
        dfc["end"] = pd.to_datetime(dfc["end"], errors="coerce")
        
        # Filter events for current month
        current_month = st.session_state.calendar_date.month
        current_year = st.session_state.calendar_date.year
        
        mask = (
            ((dfc["start"].dt.month == current_month) & (dfc["start"].dt.year == current_year)) |
            ((dfc["end"].dt.month == current_month) & (dfc["end"].dt.year == current_year))
        )
        dfv = dfc[mask].copy()
        
        if not dfv.empty:
            dfv = dfv.sort_values("start")
            show_df = dfv[["title", "start", "end", "label", "location", "description"]]
            st.dataframe(show_df, use_container_width=True)
            
            # Delete button for each event
            with st.expander("🗑️ Delete Events"):
                for idx, row in dfv.iterrows():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{row['title']}** - {row['start'].strftime('%Y-%m-%d %H:%M')}")
                    with col2:
                        if st.button("Delete", key=f"del_{idx}"):
                            st.session_state.calendar_events = [
                                e for e in st.session_state.calendar_events 
                                if e["title"] != row["title"] or e["start"] != row["start"].isoformat()
                            ]
                            # save_events(st.session_state.calendar_events)
                            st.rerun()
        else:
            st.info("No events this month")
    except Exception as e:
        st.error(f"Failed to display events: {e}")
else:
    st.info("No events yet. Use the form below to add one.")
st.markdown("---")

with st.form("new_event_form", clear_on_submit=True):
    st.subheader("Create New Event")
    colA, colB = st.columns(2)
    with colA:
        title = st.text_input("Title", max_chars=120)
        start_date = st.date_input("Start date")
        start_time = st.time_input("Start time", value=None, key="start_time")
    with colB:
        description = st.text_input("Description", value="")
        end_date = st.date_input("End date")
        end_time = st.time_input("End time", value=None, key="end_time")

    colC, colD = st.columns([1,3])
    with colC:
        location = st.text_input("Location", value="")
    with colD:
        label = st.selectbox("Label", ["meeting", "appointment", "deadline", "reminder", "other"], index=0)

    submitted = st.form_submit_button("Add Event", type="primary", use_container_width=True)
    if submitted:
        if not title:
            st.warning("Title is required")
        else:
            try:
                start_iso = _dt.combine(start_date, start_time or _dt.min.time()).isoformat()
                end_iso = _dt.combine(end_date, end_time or _dt.min.time()).isoformat()
            except Exception:
                start_iso, end_iso = str(start_date), str(end_date)

            new_event = {
                "title": title,
                "start": start_iso,
                "end": end_iso,
                "description": description,
                "location": location,
                "label": label
            }
            
            # Call backend API
            with st.spinner("Creating event..."):
                api_response = create_event_api(new_event)
            
            if api_response:
                # Add to local state and save
                st.session_state.calendar_events.append(new_event)
                # if save_events(st.session_state.calendar_events):
                #     st.success(f"✅ Event '{title}' created successfully!")
                #     st.rerun()
                # else:
                #     st.warning("⚠️ Event created via API but failed to save locally")
            else:
                st.error("Failed to create event. Please check backend connection.")

# st.divider()

# # Timeline view (full width, below calendar)
# render_timeline(events, st.session_state.calendar_date.year, st.session_state.calendar_date.month)




