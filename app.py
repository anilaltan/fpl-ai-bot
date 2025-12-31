import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data_loader import DataLoader
from src.optimizer import Optimizer
from src.simulator import MonteCarloEngine, simulate_team

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "all_players.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_custom_css() -> None:
    st.markdown(
        """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
/* App background */
.stApp {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(56, 189, 248, 0.10), transparent 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(34, 197, 94, 0.10), transparent 55%),
              linear-gradient(180deg, #0b1220 0%, #0b1220 100%);
}

/* Section titles */
.section-title {
  font-weight: 800;
  letter-spacing: 0.3px;
  margin: 0 0 8px 0;
}

/* KPI cards */
.kpi {
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.88));
  border: 1px solid rgba(148, 163, 184, 0.20);
  border-radius: 14px;
  padding: 16px 16px 14px 16px;
  box-shadow: 0 10px 40px rgba(0,0,0,0.25);
}
.kpi .label { color: rgba(226, 232, 240, 0.75); font-size: 12px; }
.kpi .value { color: #e2e8f0; font-size: 26px; font-weight: 800; margin-top: 2px; }
.kpi .sub { color: rgba(226, 232, 240, 0.70); font-size: 12px; margin-top: 6px; }

/* Player cards */
.player-card {
  background: linear-gradient(135deg, rgba(2, 6, 23, 0.95), rgba(15, 23, 42, 0.92));
  border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 10px 34px rgba(0,0,0,0.30);
  min-height: 118px;
}
.player-name {
  color: #e2e8f0;
  font-weight: 900;
  letter-spacing: 0.2px;
  font-size: 16px;
  margin: 0;
}
.player-meta {
  color: rgba(226, 232, 240, 0.70);
  font-size: 12px;
  margin-top: 2px;
}
.chip {
        display: inline-block;
  padding: 4px 8px;
  border-radius: 999px;
        font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.2px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  background: rgba(148, 163, 184, 0.10);
  color: rgba(226, 232, 240, 0.86);
}
.chip-cap { border-color: rgba(34, 197, 94, 0.55); background: rgba(34, 197, 94, 0.12); }
.chip-vc  { border-color: rgba(251, 191, 36, 0.55); background: rgba(251, 191, 36, 0.12); }
.stat-row {
  margin-top: 10px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.stat {
  background: rgba(30, 41, 59, 0.55);
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 12px;
  padding: 8px 10px;
  min-width: 88px;
}
.stat .k { color: rgba(226, 232, 240, 0.65); font-size: 10px; font-weight: 800; letter-spacing: 0.2px; }
.stat .v { color: #e2e8f0; font-size: 14px; font-weight: 900; margin-top: 2px; }
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_system() -> Tuple[DataLoader, Optimizer]:
    loader = DataLoader()
    opt = Optimizer()
    return loader, opt


@st.cache_data(ttl=60 * 10)
def load_players_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data(ttl=60 * 10)
def load_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        import json

        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=60 * 10)
def load_credentials() -> Dict[str, Dict[str, Any]]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        users = (((payload.get("credentials") or {}).get("usernames")) or {})
        if not isinstance(users, dict):
            return {}
        return users
    except Exception:
        return {}


def _verify_password(plain: str, stored_hash: str) -> bool:
    """
    Verify password using bcrypt when possible, else fall back to plain match.
    (Config uses bcrypt hashes; this fallback is only for dev convenience.)
    """
    plain = str(plain or "")
    stored_hash = str(stored_hash or "")
    if not plain or not stored_hash:
        return False
    try:
        import bcrypt  # type: ignore

        return bool(bcrypt.checkpw(plain.encode("utf-8"), stored_hash.encode("utf-8")))
    except Exception:
        return plain == stored_hash


def _init_session() -> None:
    st.session_state.setdefault("logged_in", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("user_role", "free")
    st.session_state.setdefault("display_name", "")


def _login_gate() -> None:
    """
    Simple login screen. On success, stores user_role in st.session_state.
    """
    _init_session()
    if st.session_state.get("logged_in"):
        return

    users = load_credentials()
    st.set_page_config(page_title="FPL AI Architect", layout="wide")
    _load_custom_css()

    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("## 🔐 Login")
        st.caption("Access your FPL management panel.")

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="e.g. admin / premium_user / testuser")
            password = st.text_input("Password", type="password", placeholder="Your password")
            submit = st.form_submit_button("Sign in", type="primary")

        if submit:
            u = users.get(username) if isinstance(users, dict) else None
            if not u:
                st.error("Invalid username or password.")
                st.stop()

            stored = str(u.get("password", ""))
            if not _verify_password(password, stored):
                st.error("Invalid username or password.")
                st.stop()

            role = str(u.get("role", "free") or "free").lower()
            name = str(u.get("name", username) or username)

            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["user_role"] = role
            st.session_state["display_name"] = name
            st.rerun()

        st.info("Demo users are defined in `config.yaml` (credentials.usernames).")

    st.stop()


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _render_kpi(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
<div class="kpi">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="sub">{sub}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_player_card(row: pd.Series, badge: str = "") -> None:
    name = str(row.get("web_name", ""))
    team = str(row.get("team_name", ""))
    pos = str(row.get("position", ""))
    xp = float(row.get("gw19_xP", 0.0) or 0.0)
    spt = float(row.get("set_piece_threat", 0.0) or 0.0)
    ma = float(row.get("matchup_advantage", 1.0) or 1.0)

    # Risk yönetimi: availability kontrolü
    availability_score = float(row.get("availability_score", 1.0) or 1.0)
    news = str(row.get("news", "")).strip()
    risk_warning = ""

    if availability_score < 1.0:
        # Risk seviyesine göre icon rengi
        if availability_score < 0.5:
            icon_color = "#ef4444"  # Kırmızı - yüksek risk
        else:
            icon_color = "#f59e0b"  # Turuncu - orta risk

        # Tooltip için haber metni hazırla
        tooltip_text = news if news else f"Availability: {availability_score:.1%}"
        risk_warning = f'<span title="{tooltip_text}" style="color: {icon_color}; margin-left: 8px;">⚠️</span>'

    badges = []
    if badge == "C":
        badges.append('<span class="chip chip-cap">CAPTAIN</span>')
    elif badge == "VC":
        badges.append('<span class="chip chip-vc">VICE</span>')
    badges_html = "&nbsp;".join(badges)

    st.markdown(
        f"""
    <div class="player-card">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
            <div>
      <p class="player-name">{name}{risk_warning}</p>
      <div class="player-meta">{team} • {pos}</div>
            </div>
    <div>{badges_html}</div>
            </div>
  <div class="stat-row">
    <div class="stat"><div class="k">GW19 xP</div><div class="v">{xp:.2f}</div></div>
    <div class="stat"><div class="k">Risk Adj</div><div class="v">{availability_score:.1%}</div></div>
    <div class="stat"><div class="k">Set Piece</div><div class="v">{spt:.2f}</div></div>
        </div>
    </div>
        """,
        unsafe_allow_html=True,
    )


def _choose_target_metric(df: pd.DataFrame, gw_id: Optional[int]) -> str:
    """
    Prefer a gw{N}_xP column if present; otherwise default to gw19_xP.
    """
    if gw_id is not None:
        candidate = f"gw{int(gw_id)}_xP"
        if candidate in df.columns:
            return candidate
    if "gw19_xP" in df.columns:
        return "gw19_xP"
    # last resort: any gw*_xP
    gw_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("gw") and c.endswith("_xP")]
    return gw_cols[0] if gw_cols else "gw19_xP"


def _role_can_transfer(role: str) -> bool:
    role = str(role or "free").lower()
    return role in ("premium", "admin")


def main() -> None:
    _login_gate()
    st.set_page_config(page_title="FPL AI Architect", layout="wide")
    _load_custom_css()

    loader, opt = get_system()
    df_players = load_players_df()
    meta = load_metadata()

    # Dynamic GW context (prefer metadata saved by updater; else API)
    gw_id: Optional[int] = None
    gw_name: str = ""
    if meta:
        try:
            gw_id = int(meta.get("id")) if meta.get("id") is not None else None
            gw_name = str(meta.get("name") or "")
        except Exception:
            gw_id = None
            gw_name = ""
    if not gw_id:
        try:
            gw_info = loader.get_next_gw()
            gw_id = int(gw_info.get("id") or 0) or None
            gw_name = str(gw_info.get("name") or "")
        except Exception:
            gw_id = None
            gw_name = ""

    user_role = str(st.session_state.get("user_role", "free"))
    display_name = str(st.session_state.get("display_name", st.session_state.get("username", "")))

    top = st.columns([1, 1])
    with top[0]:
        st.markdown("## FPL AI Architect")
        if gw_id:
            st.caption(f"Target: {gw_name or 'Next'} (GW{gw_id}) • Logged in as **{display_name}** ({user_role})")
        else:
            st.caption(f"Logged in as **{display_name}** ({user_role})")
    with top[1]:
        st.write("")
        st.write("")
        if st.button("Logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["display_name"] = ""
            st.session_state["user_role"] = "free"
            st.rerun()

    # --- SIDEBAR ---
    st.sidebar.title("⚙️ Strategy Settings")
    budget = st.sidebar.slider("Budget (£)", 80.0, 105.0, 100.0, step=0.1)
    risk_mode = st.sidebar.selectbox("Risk Tolerance", ["Conservative", "Aggressive"])
    run_btn = st.sidebar.button("Generate Squad", type="primary")

    if df_players.empty:
        st.error(
            "Player dataset not found. Please generate `data/all_players.csv` by running `updater.py`."
        )
        st.stop()

    required = [
        "web_name",
        "team_name",
        "position",
        "price",
        "gw19_xP",
        "risk_multiplier",
        "set_piece_threat",
        "matchup_advantage",
    ]
    missing = [c for c in required if c not in df_players.columns]
    if missing:
        st.error(f"Dataset schema missing required columns: {missing}")
        st.stop()

    df_players = _coerce_numeric(
        df_players,
        ["price", "gw19_xP", "risk_multiplier", "set_piece_threat", "matchup_advantage", "selected_by_percent"],
    ).fillna(0.0)

    metric_col = _choose_target_metric(df_players, gw_id)
    if metric_col not in df_players.columns:
        metric_col = "gw19_xP"

    # Normalize ownership (selected_by_percent) to float
    if "selected_by_percent" in df_players.columns:
        df_players["selected_by_percent"] = (
            df_players["selected_by_percent"]
            .astype(str)
            .str.replace("%", "", regex=False)
        )
        df_players["selected_by_percent"] = pd.to_numeric(df_players["selected_by_percent"], errors="coerce").fillna(0.0)

    # Risk modes: adjust optimization target without mutating underlying metric display
    df_opt = df_players.copy()
    rm = df_opt["risk_multiplier"].replace(0, 1.0).clip(lower=0.01, upper=1.0)
    if risk_mode == "Conservative":
        df_opt["xP_opt"] = df_opt[metric_col] * rm
        risk_sub = "Extra risk penalty applied (safer minutes)."
    else:
        df_opt["xP_opt"] = df_opt[metric_col] / rm
        risk_sub = "Risk penalty relaxed (higher ceiling)."

    # Precompute a squad when the button is clicked, reuse across tabs
    if run_btn:
        with st.spinner("Optimizing squad..."):
            squad = opt.solve_dream_team(df_opt, target_metric="xP_opt", budget=float(budget))
        st.session_state["last_squad"] = squad
    else:
        squad = st.session_state.get("last_squad")

    tab_labels = ["Dream Team", "Transfer Wizard", "Player Explorer", "Chip Strategy", "Simulation Lab"]
    tabs = st.tabs(tab_labels)

    # -------------------- TAB 1: Dream Team (Free) -------------------- #
    with tabs[0]:
        st.markdown("### Dream Team")
        st.caption("Pitch view + Alpha matrix. Captaincy suggestions: safe / differential / vice.")

        # Ensemble Model Legend
        with st.expander("🎯 Ensemble Model Açıklaması", expanded=False):
            st.markdown("""
            **Mixture of Experts (3 Uzman Model):**
            - **Tech (50% 🟦)**: xG, xA, Form - Geleneksel istatistikler
            - **Mkt (30% 🟩)**: Bahis oranları - Piyasa zekası
            - **Tact (20% 🟥)**: Eşleşme + Duran top - Kısa vadeli taktik

            **Final xP = (Tech × 0.5) + (Mkt × 0.3) + (Tact × 0.2)**
            """)

        if squad is None or getattr(squad, "empty", True):
            st.info("Click **Generate Squad** in the sidebar to build your Dream Team.")
            st.stop()

        squad = squad.copy()
        # Ensure display columns exist
        for col in [metric_col, "set_piece_threat", "matchup_advantage", "price", "selected_by_percent"]:
            if col not in squad.columns and col in df_players.columns and "id" in squad.columns:
                # best-effort merge by id
                try:
                    squad = squad.merge(
                        df_players[["id", col]], on="id", how="left", suffixes=("", "_src")
                    )
                except Exception:
                    pass

        # Display metric
        if metric_col not in squad.columns:
            # fall back if optimizer ran on xP_opt only
            squad[metric_col] = squad["xP_opt"] if "xP_opt" in squad.columns else 0.0

        squad = _coerce_numeric(
            squad, ["price", metric_col, "set_piece_threat", "matchup_advantage", "selected_by_percent"]
        ).fillna(0.0)

        total_xp = float(squad[metric_col].sum())
        total_cost = float(squad["price"].sum())
        bank = float(budget) - total_cost

        # Captaincy suggestions
        sorted_sq = squad.sort_values(metric_col, ascending=False).reset_index(drop=True)
        safe_pick = sorted_sq.iloc[0] if len(sorted_sq) >= 1 else None
        vice_pick = sorted_sq.iloc[1] if len(sorted_sq) >= 2 else None
        diff_pool = sorted_sq[sorted_sq.get("selected_by_percent", 0.0) < 10.0]
        diff_pick = diff_pool.iloc[0] if not diff_pool.empty else None

        k1, k2, k3 = st.columns(3)
        with k1:
            _render_kpi(
                f"Projected Points ({metric_col})",
                f"{total_xp:.2f}",
                risk_sub,
            )
        with k2:
            _render_kpi("Total Cost", f"£{total_cost:.1f}", "Optimized 15-player squad")
        with k3:
            _render_kpi("Bank", f"£{bank:.1f}", f"Budget: £{budget:.1f}")

        st.markdown("---")
        st.markdown("### Captaincy Radar")
        c1, c2, c3 = st.columns(3)
        with c1:
            if safe_pick is not None:
                st.success(
                    f"🛡️ **Safe Pick**: {safe_pick.get('web_name')} "
                    f"({metric_col}={float(safe_pick.get(metric_col, 0.0)):.2f})"
                )
        with c2:
            if diff_pick is not None:
                st.info(
                    f"🚀 **Differential** (<10% owned): {diff_pick.get('web_name')} "
                    f"({metric_col}={float(diff_pick.get(metric_col, 0.0)):.2f}, "
                    f"owned={float(diff_pick.get('selected_by_percent', 0.0)):.1f}%)"
                )
            else:
                st.warning("🚀 Differential: No <10% owned player found in this squad.")
        with c3:
            if vice_pick is not None:
                st.warning(
                    f"⚔️ **Vice Captain**: {vice_pick.get('web_name')} "
                    f"({metric_col}={float(vice_pick.get(metric_col, 0.0)):.2f})"
                )

        st.markdown("---")
        st.markdown("### Visual Pitch")
        pos_order = ["GK", "DEF", "MID", "FWD"]

        captain_name = str(safe_pick.get("web_name")) if safe_pick is not None else ""
        vice_name = str(vice_pick.get("web_name")) if vice_pick is not None else ""

        for pos in pos_order:
            st.markdown(f"**{pos} Line**")
            row_df = squad[squad["position"].astype(str).str.upper() == pos].copy()
            row_df = row_df.sort_values(metric_col, ascending=False)
            if row_df.empty:
                st.info(f"No players in {pos} (unexpected).")
                continue
            cols = st.columns(len(row_df))
            for i, (_, p) in enumerate(row_df.iterrows()):
                badge = ""
                if captain_name and str(p.get("web_name", "")) == captain_name:
                    badge = "C"
                elif vice_name and str(p.get("web_name", "")) == vice_name:
                    badge = "VC"
                with cols[i]:
                    # ensure _render_player_card reads gw19_xP; patch value in row for display
                    p = p.copy()
                    p["gw19_xP"] = float(p.get(metric_col, 0.0) or 0.0)
                    _render_player_card(p, badge=badge)

        st.markdown("---")
        st.markdown("### 📊 Alpha Intelligence Matrix")
        display_cols = [
            "web_name",
            "team_name",
            "position",
            "price",
            metric_col,
            "technical_score",
            "market_score",
            "tactical_score",
            "set_piece_threat",
            "matchup_advantage",
            "selected_by_percent",
        ]
        display_cols = [c for c in display_cols if c in squad.columns]
        table_df = squad[display_cols].copy()
        table_df = table_df.sort_values(["position", metric_col], ascending=[True, False])

        # Define metric hint for display
        metric_hint = "final_5gw_xP" if "final_5gw_xP" in squad.columns else "long_term_xP"

        # Column configuration for ensemble scores
        column_config = {
            metric_col: st.column_config.NumberColumn(
                f"Final xP ({metric_hint})",
                help=f"Ensemble score: Tech(50%) + Mkt(30%) + Tact(20%)",
                format="%.2f"
            ),
            "technical_score": st.column_config.ProgressColumn(
                "Tech (50%)",
                help="Technical Score: xG, xA, Form based",
                format="%.2f",
                min_value=0,
                max_value=15,
            ),
            "market_score": st.column_config.ProgressColumn(
                "Mkt (30%)",
                help="Market Score: Betting odds based",
                format="%.2f",
                min_value=0,
                max_value=15,
            ),
            "tactical_score": st.column_config.ProgressColumn(
                "Tact (20%)",
                help="Tactical Score: Matchup + Set pieces",
                format="%.2f",
                min_value=0,
                max_value=15,
            ),
            "set_piece_threat": st.column_config.NumberColumn(
                "Set Piece",
                format="%.2f"
            ),
            "matchup_advantage": st.column_config.NumberColumn(
                "Matchup",
                format="%.2f"
            ),
            "price": st.column_config.NumberColumn(
                "£",
                format="%.1f"
            )
        }

        # Legacy styling for backward compatibility
        styled = (
            table_df.style.background_gradient(subset=[metric_col], cmap="Greens")
            .background_gradient(subset=["set_piece_threat"], cmap="Blues")
            .background_gradient(subset=["matchup_advantage"], cmap="Purples")
        )

        st.dataframe(
            styled,
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )

    # -------------------- TAB 2: Transfer Wizard (Premium/Admin) -------------------- #
    with tabs[1]:
        st.markdown("### Transfer Wizard")
        if not _role_can_transfer(user_role):
            st.warning("This feature is **Premium/Admin only**. Upgrade to access Transfer Wizard.")
            st.stop()

        st.caption("Pick your current 15-man squad and let the optimizer suggest the best single transfer.")

        # Ensemble Model Legend
        with st.expander("🎯 Ensemble Model Açıklaması", expanded=False):
            st.markdown("""
            **Mixture of Experts (3 Uzman Model):**
            - **Tech (50% 🟦)**: xG, xA, Form - Geleneksel istatistikler
            - **Mkt (30% 🟩)**: Bahis oranları - Piyasa zekası
            - **Tact (20% 🟥)**: Eşleşme + Duran top - Kısa vadeli taktik

            **Final xP = (Tech × 0.5) + (Mkt × 0.3) + (Tact × 0.2)**
            """)

        # Initialize session state for team defaults
        if 'my_team_defaults' not in st.session_state:
            st.session_state['my_team_defaults'] = []

        # Auto-fetch team section
        st.markdown("#### 📥 Takım İçe Aktar")
        col1, col2 = st.columns([3, 1])

        with col1:
            team_id_input = st.text_input(
                "FPL Team ID",
                placeholder="Örn: 123456",
                key="team_id_input"
            )

        with col2:
            fetch_team = st.button("📥 Takımı Getir", type="secondary")

        if fetch_team and team_id_input.strip():
            with st.spinner("Takım çekiliyor..."):
                team_names = loader.fetch_user_team(team_id_input.strip(), df_players)
                if team_names:
                    st.session_state['my_team_defaults'] = team_names
                    st.success(f"✅ {len(team_names)} oyuncu başarıyla çekildi!")
                    st.rerun()  # Refresh to update multiselect
                else:
                    st.error("❌ Takım çekilemedi. Team ID'yi kontrol edin.")

        st.info("💡 Not: FPL gizlilik kuralları gereği son 'kesinleşmiş' kadronuz çekildi. Bu hafta transfer yaptıysanız lütfen listeyi güncelleyin.")

        # Build selection list
        if "id" not in df_players.columns:
            st.error("Dataset missing 'id' column required for transfer wizard.")
            st.stop()

        pool = df_players.copy()
        pool["label"] = (
            pool["web_name"].astype(str)
            + " • "
            + pool["team_name"].astype(str)
            + " • "
            + pool["position"].astype(str)
            + " • £"
            + pool["price"].astype(float).map(lambda x: f"{x:.1f}")
        )
        label_to_id = dict(zip(pool["label"].tolist(), pool["id"].astype(int).tolist()))
        options = sorted(label_to_id.keys())

        # Create reverse mapping for defaults
        name_to_label = {row['web_name']: label for label, pid in label_to_id.items()
                        for _, row in pool.iterrows() if row['id'] == pid}
        default_labels = [name_to_label.get(name, "") for name in st.session_state['my_team_defaults']]
        default_labels = [label for label in default_labels if label in options]

        selected = st.multiselect(
            "Select your current squad (max 15)",
            options=options,
            default=default_labels,
            max_selections=15,
        )
        bank_balance = st.number_input("Bank Balance (£m)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

        # Number of suggestions slider
        num_suggestions = st.slider("Öneri Sayısı (Top N)", min_value=1, max_value=10, value=3)

        analyze = st.button("Analyze Transfer", type="primary")

        if analyze:
            if len(selected) == 0:
                st.error("Select at least 1 player from your squad.")
                st.stop()

            ids = [label_to_id[x] for x in selected]
            current_team_df = df_players[df_players["id"].isin(ids)].copy()

            suggestions = opt.suggest_transfer(current_team_df, df_players, float(bank_balance), num_suggestions)
            if not suggestions:
                st.info("No beneficial transfer found within your budget.")
                st.stop()

            metric_hint = "final_5gw_xP" if "final_5gw_xP" in df_players.columns else "long_term_xP"

            st.markdown("#### Transfer Recommendations")

            for i, suggestion in enumerate(suggestions, 1):
                p_out = suggestion["out"]
                p_in = suggestion["in"]
                gain = float(suggestion.get("gain", 0.0) or 0.0)
                remaining_bank = float(suggestion.get("remaining_bank", 0.0) or 0.0)

                out_xp = float(p_out.get(metric_hint, 0.0) or 0.0)
                in_xp = float(p_in.get(metric_hint, 0.0) or 0.0)
                in_ma = float(p_in.get("matchup_advantage", 1.0) or 1.0)

                with st.expander(f"**Option #{i}:** 🔴 Sat: {p_out.get('web_name')} → 🟢 Al: {p_in.get('web_name')} | ⚡ Gain: +{gain:.2f}", expanded=(i==1)):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.error(
                            f"🔻 **OUT**: {p_out.get('web_name')} "
                            f"(£{float(p_out.get('price', 0.0) or 0.0):.1f}, {metric_hint}={out_xp:.2f})"
                        )

                    with col2:
                        st.success(
                            f"🟢 **IN**: {p_in.get('web_name')} "
                            f"(£{float(p_in.get('price', 0.0) or 0.0):.1f}, {metric_hint}={in_xp:.2f}, "
                            f"matchup_adv={in_ma:.2f})"
                        )

                    with col3:
                        st.metric("📈 Net Gain", f"{gain:+.2f}")
                        st.metric("💰 Remaining Bank", f"£{remaining_bank:.1f}")

    # -------------------- TAB 3: Player Explorer -------------------- #
    with tabs[2]:
        st.markdown("### Player Explorer")
        st.caption("Search / filter the player pool. Includes set_piece_threat and matchup_advantage.")

        df_exp = df_players.copy()

        # Filters
        f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1])
        with f1:
            query = st.text_input("Search (name/team)", value="")
        with f2:
            pos_sel = st.multiselect("Position", options=["GK", "DEF", "MID", "FWD"], default=[])
        with f3:
            teams = sorted(df_exp["team_name"].astype(str).unique().tolist())
            team_sel = st.multiselect("Team", options=teams, default=[])
        with f4:
            min_mins = st.number_input("Min Minutes", min_value=0, max_value=4000, value=0, step=50)

        if query.strip():
            q = query.strip().lower()
            df_exp = df_exp[
                df_exp["web_name"].astype(str).str.lower().str.contains(q)
                | df_exp["team_name"].astype(str).str.lower().str.contains(q)
            ].copy()
        if pos_sel:
            df_exp = df_exp[
                df_exp["position"].astype(str).str.upper().isin([p.upper() for p in pos_sel])
            ].copy()
        if team_sel:
            df_exp = df_exp[df_exp["team_name"].astype(str).isin(team_sel)].copy()
        if "minutes" in df_exp.columns:
            df_exp["minutes"] = pd.to_numeric(df_exp["minutes"], errors="coerce").fillna(0.0)
            df_exp = df_exp[df_exp["minutes"] >= float(min_mins)].copy()

        sort_cols = [
            metric_col,
            "price",
            "set_piece_threat",
            "matchup_advantage",
            "selected_by_percent",
        ]
        sort_cols = [c for c in sort_cols if c in df_exp.columns]
        sort_by = st.selectbox("Sort by", options=sort_cols, index=0 if sort_cols else 0)
        asc = st.checkbox("Ascending", value=False)
        if sort_by:
            df_exp = df_exp.sort_values(sort_by, ascending=asc)

        show_cols = [
            "web_name",
            "team_name",
            "position",
            "price",
            metric_col,
            "technical_score",
            "market_score",
            "tactical_score",
            "set_piece_threat",
            "matchup_advantage",
            "selected_by_percent",
            "minutes",
        ]
        show_cols = [c for c in show_cols if c in df_exp.columns]

        # Column configuration for ensemble scores
        column_config = {
            metric_col: st.column_config.NumberColumn(
                f"Final xP ({metric_hint})",
                help=f"Ensemble score: Tech(50%) + Mkt(30%) + Tact(20%)",
                format="%.2f"
            ),
            "technical_score": st.column_config.ProgressColumn(
                "Tech (50%)",
                help="Technical Score: xG, xA, Form based",
                format="%.2f",
                min_value=0,
                max_value=15,
            ),
            "market_score": st.column_config.ProgressColumn(
                "Mkt (30%)",
                help="Market Score: Betting odds based",
                format="%.2f",
                min_value=0,
                max_value=15,
            ),
            "tactical_score": st.column_config.ProgressColumn(
                "Tact (20%)",
                help="Tactical Score: Matchup + Set pieces",
                format="%.2f",
                min_value=0,
                max_value=15,
            ),
            "set_piece_threat": st.column_config.NumberColumn(
                "Set Piece",
                format="%.2f"
            ),
            "matchup_advantage": st.column_config.NumberColumn(
                "Matchup",
                format="%.2f"
            ),
            "price": st.column_config.NumberColumn(
                "£",
                format="%.1f"
            )
        }

        # Legacy styling for backward compatibility
        styled = (
            df_exp[show_cols]
            .head(250)
            .style.background_gradient(subset=[metric_col], cmap="Greens")
            .background_gradient(subset=["set_piece_threat"], cmap="Blues")
            .background_gradient(subset=["matchup_advantage"], cmap="Purples")
        )

        st.dataframe(
            styled,
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )

    # -------------------- TAB 4: Chip Strategy --------------------
    with tabs[3]:
        st.markdown("### Chip Strategy Advisor")
        st.caption("FPL chip'lerinizi (Wildcard, Triple Captain, Bench Boost) ne zaman kullanacağınızı analiz eder.")

        # Gameweek selector
        current_gw = st.selectbox(
            "Analiz Edilecek Gameweek",
            options=list(range(1, 39)),
            index=18,  # Default to GW19
            help="Hangi haftayı analiz etmek istiyorsunuz?"
        )

        # Run analysis button
        analyze_chips = st.button("🔍 Chip'leri Analiz Et", type="primary")

        if analyze_chips:
            with st.spinner("Chip stratejileri analiz ediliyor..."):
                # Get user team from session state or show message
                if 'my_team_defaults' not in st.session_state or not st.session_state['my_team_defaults']:
                    st.error("⚠️ Önce Transfer Wizard'dan takımınızı seçin veya Team ID ile içe aktarın!")
                    st.stop()

                # Get selected players
                selected_names = st.session_state['my_team_defaults']
                if not selected_names:
                    st.error("Takım bulunamadı!")
                    st.stop()

                # Get player data
                team_df = df_players[df_players['web_name'].isin(selected_names)].copy()
                if team_df.empty:
                    st.error("Seçilen oyuncular bulunamadı!")
                    st.stop()

                # Analyze chips
                chip_analysis = opt.analyze_chips(team_df, df_players, current_gw)

                # Display results
                st.markdown("#### 📊 Chip Analiz Sonuçları")

                # Create 4 columns for chips
                col1, col2, col3, col4 = st.columns(4)

                chips = [
                    ('WC', 'Wildcard', '🎭', col1),
                    ('TC', 'Triple Captain', '👑', col2),
                    ('BB', 'Bench Boost', '🪑', col3),
                    ('FH', 'Free Hit', '🎯', col4)
                ]

                for chip_key, chip_name, emoji, col in chips:
                    with col:
                        if chip_key in chip_analysis:
                            analysis = chip_analysis[chip_key]
                            status = analysis['status']
                            score = analysis['score']
                            reason = analysis['reason']

                            # Color coding
                            if 'Öneriliyor' in status:
                                color = '🟢'
                                bg_color = '#d4edda'
                            elif 'Düşünülebilir' in status or 'İzlemede' in status:
                                color = '🟡'
                                bg_color = '#fff3cd'
                            else:
                                color = '⚪'
                                bg_color = '#f8f9fa'

                            st.markdown(
                                f"""
                                <div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; text-align: center;">
                                    <h4>{emoji} {chip_name}</h4>
                                    <h3 style="color: {'green' if color == '🟢' else 'orange' if color == '🟡' else 'gray'};">{status}</h3>
                                    <p style="font-size: 12px;">{reason}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:
                            # Free Hit placeholder
                            st.markdown(
                                f"""
                                <div style="background-color: #e9ecef; padding: 10px; border-radius: 5px; text-align: center;">
                                    <h4>{emoji} {chip_name}</h4>
                                    <h3 style="color: gray;">Yakında</h3>
                                    <p style="font-size: 12px;">Bu chip henüz analiz edilmiyor</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                # Additional insights
                st.markdown("---")
                st.markdown("#### 💡 Öneriler")

                wc_status = chip_analysis.get('WC', {}).get('status', '')
                tc_status = chip_analysis.get('TC', {}).get('status', '')
                bb_status = chip_analysis.get('BB', {}).get('status', '')

                insights = []

                if 'Öneriliyor' in wc_status:
                    insights.append("🎭 **Wildcard Zamanı!** Kadronuzun verimi düşük, büyük değişiklikler yapın.")
                elif 'Öneriliyor' in tc_status:
                    insights.append("👑 **Triple Captain Şansı!** Yıldız oyuncunuz bu hafta parlayabilir.")
                elif 'Öneriliyor' in bb_status:
                    insights.append("🪑 **Bench Boost Uygun!** Yedekleriniz starter'lardan daha iyi performans gösterebilir.")

                if not insights:
                    insights.append("✅ **Şu Anda Chip Gerekmiyor.** Mevcut kadronuzla devam edin.")

                for insight in insights:
                    st.info(insight)

    # -------------------- TAB 5: Simulation Lab -------------------- #
    with tabs[4]:
        st.markdown("### 🎲 Simulation Lab")
        st.caption("Monte Carlo simülasyonları ile takımınızın olasılıksal performansını analiz edin.")

        # Check if we have player data
        if df_players is None or df_players.empty:
            st.error("Oyuncu verisi bulunamadı! Önce verileri yükleyin.")
            st.stop()

        # Get current team from session state
        current_team = st.session_state.get('my_team_defaults', [])
        if not current_team:
            st.info("💡 Transfer Wizard sekmesinden takımınızı seçin veya Team ID ile içe aktarın.")
            st.markdown("**Alternatif:** Aşağıdan rastgele bir Dream Team seçebilirsiniz.")

            # Option to use dream team for simulation
            use_dream_team = st.checkbox("Dream Team ile simülasyon yap")
            if use_dream_team and squad is not None and not squad.empty:
                current_team = squad['web_name'].tolist()
                st.success(f"Dream Team seçildi: {len(current_team)} oyuncu")
            else:
                st.stop()

        # Get team data
        team_df = df_players[df_players['web_name'].isin(current_team)].copy()
        if team_df.empty:
            st.error("Seçilen takım oyuncuları bulunamadı!")
            st.stop()

        st.markdown(f"**Simüle Edilen Takım:** {len(team_df)} oyuncu")

        # Simulation parameters
        col1, col2 = st.columns(2)
        with col1:
            n_simulations = st.slider(
                "Simülasyon Sayısı",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                help="Daha fazla simülasyon = daha doğru sonuçlar (ama daha yavaş)"
            )

        with col2:
            n_gameweeks = st.slider(
                "Gameweek Sayısı",
                min_value=1,
                max_value=10,
                value=5,
                help="Kaç haftalık performans simüle edilsin?"
            )

        # Captain selection
        captain_options = ["Otomatik"] + sorted(team_df['web_name'].tolist())
        captain_name = st.selectbox(
            "Kaptan Seçin",
            options=captain_options,
            index=0,
            help="Otomatik seçilirse en yüksek xP'ye sahip oyuncu kaptan yapılır"
        )

        if captain_name == "Otomatik":
            captain_name = team_df.loc[team_df['gw19_xP'].idxmax(), 'web_name']

        # Run simulation button
        run_simulation = st.button("🚀 Simülasyonu Başlat", type="primary")

        if run_simulation:
            with st.spinner(f"{n_simulations} simülasyon çalışıyor..."):
                # Create simulation engine
                engine = MonteCarloEngine(
                    n_simulations=n_simulations,
                    n_gameweeks=n_gameweeks,
                    random_seed=42  # For reproducible results
                )

                # Run simulation
                sim_result = engine.simulate_team_performance(
                    team_df=team_df,
                    captain_name=captain_name
                )

            # Display results
            st.markdown("---")
            st.markdown("## 📊 Takım Performans Dağılımı")

            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(sim_result.team_total_points, bins=50, alpha=0.7, color='#38bdf8', edgecolor='black')
            ax.axvline(sim_result.risk_metrics['mean'], color='red', linestyle='--', linewidth=2, label=f'Ortalama: {sim_result.risk_metrics["mean"]:.1f}')
            ax.axvline(sim_result.risk_metrics['var_95'], color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {sim_result.risk_metrics["var_95"]:.1f}')
            ax.axvline(sim_result.risk_metrics['p95'], color='green', linestyle='--', linewidth=2, label=f'Tavan (P95): {sim_result.risk_metrics["p95"]:.1f}')

            ax.set_xlabel('Toplam Puan')
            ax.set_ylabel('Frekans')
            ax.set_title(f'Takım Performans Dağılımı ({n_simulations} Simülasyon)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # Risk metrics display
            st.markdown("### 📈 Risk Metrikleri")

            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

            with metrics_col1:
                st.metric(
                    "Ortalama Beklenti",
                    f"{sim_result.risk_metrics['mean']:.1f}",
                    help="Beklenen ortalama puan"
                )

            with metrics_col2:
                st.metric(
                    "Güvenli Puan (VaR 95%)",
                    f"{sim_result.risk_metrics['var_95']:.1f}",
                    help="95% olasılıkla en az bu kadar puan alırsınız"
                )

            with metrics_col3:
                st.metric(
                    "Tavan Puan (Top 5%)",
                    f"{sim_result.risk_metrics['p95']:.1f}",
                    help="En iyi 5%'lik senaryoda alacağınız puan"
                )

            with metrics_col4:
                st.metric(
                    "Mega Haul İhtimali",
                    f"{sim_result.risk_metrics['prob_mega_week']:.1%}",
                    help="Çok yüksek puan (>15/GW) alma ihtimali"
                )

            # Additional insights
            st.markdown("---")
            st.markdown("### 💡 Risk Analizi")

            risk_level = "Düşük" if sim_result.risk_metrics['std'] < 10 else "Orta" if sim_result.risk_metrics['std'] < 20 else "Yüksek"

            st.info(f"""
            **Risk Seviyesi: {risk_level}**
            - Standart Sapma: {sim_result.risk_metrics['std']:.1f}
            - Tutarlılık Skoru: {sim_result.risk_metrics['consistency_score']:.2f}
            - Konservatif Tahmin (P10): {sim_result.risk_metrics['p10']:.1f}
            - Optimistik Tahmin (P90): {sim_result.risk_metrics['p90']:.1f}
            """)

        # Captaincy Duel Section
        st.markdown("---")
        st.markdown("## ⚔️ Kaptanlık Düellosu")
        st.caption("İki oyuncuyu karşılaştırın ve kaptanlık kararınızı olasılıksal analizle destekleyin.")

        # Duel simulation parameters (independent from main simulation)
        duel_n_simulations = st.slider(
            "Düello Simülasyon Sayısı",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            key="duel_simulations",
            help="Düello için simülasyon sayısı"
        )

        duel_n_gameweeks = st.slider(
            "Düello Gameweek Sayısı",
            min_value=1,
            max_value=5,
            value=3,
            key="duel_gameweeks",
            help="Düello için haftalık simülasyon sayısı"
        )

        # Player selection
        duel_col1, duel_col2 = st.columns(2)

        with duel_col1:
            player1_options = ["Seçiniz..."] + sorted(team_df['web_name'].tolist())
            player1_name = st.selectbox(
                "1. Oyuncu",
                options=player1_options,
                key="duel_player1"
            )

        with duel_col2:
            player2_options = ["Seçiniz..."] + sorted(team_df['web_name'].tolist())
            player2_name = st.selectbox(
                "2. Oyuncu",
                options=player2_options,
                key="duel_player2"
            )

        # Run duel analysis
        if player1_name != "Seçiniz..." and player2_name != "Seçiniz..." and player1_name != player2_name:
            duel_button = st.button("⚔️ Düello Analizini Başlat", type="secondary")

            if duel_button:
                with st.spinner("Oyuncular karşılaştırılıyor..."):
                    # Get player data
                    p1_data = team_df[team_df['web_name'] == player1_name].iloc[0]
                    p2_data = team_df[team_df['web_name'] == player2_name].iloc[0]

                    player1_info = {
                        'name': player1_name,
                        'xp': float(p1_data.get('gw19_xP', 0.0) or 0.0),
                        'position': str(p1_data.get('position', 'MID'))
                    }

                    player2_info = {
                        'name': player2_name,
                        'xp': float(p2_data.get('gw19_xP', 0.0) or 0.0),
                        'position': str(p2_data.get('position', 'MID'))
                    }

                    # Create engine for duel analysis
                    duel_engine = MonteCarloEngine(
                        n_simulations=duel_n_simulations,
                        n_gameweeks=duel_n_gameweeks,
                        random_seed=42
                    )

                    # Run comparison
                    duel_simulations = duel_engine.compare_players(player1_info, player2_info, duel_n_gameweeks)
                    duel_analysis = duel_engine.analyze_captaincy_duel(
                        duel_simulations[player1_name],
                        duel_simulations[player2_name],
                        player1_name,
                        player2_name
                    )

                # Display results
                st.markdown("### 📊 Düello Sonuçları")

                # Create overlapping density plot
                fig, ax = plt.subplots(figsize=(12, 6))

                # Captaincy points (2x multiplier)
                p1_captaincy = duel_simulations[player1_name] * 2
                p2_captaincy = duel_simulations[player2_name] * 2

                # Create density plots
                sns.kdeplot(data=p1_captaincy, fill=True, alpha=0.6, color='#ef4444',
                           label=f'{player1_name} (Kaptan)', ax=ax)
                sns.kdeplot(data=p2_captaincy, fill=True, alpha=0.6, color='#3b82f6',
                           label=f'{player2_name} (Kaptan)', ax=ax)

                ax.set_xlabel('Kaptanlık Puanları (2x çarpan ile)')
                ax.set_ylabel('Yoğunluk')
                ax.set_title(f'{player1_name} vs {player2_name} - Kaptanlık Performans Dağılımı')
                ax.legend()
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

                # Duel statistics
                st.markdown("### 🏆 Karşılaştırma Metrikleri")

                stats_col1, stats_col2, stats_col3 = st.columns(3)

                with stats_col1:
                    st.metric(
                        f"{player1_name} Ortalama",
                        f"{duel_analysis[player1_name]['mean_captaincy_points']:.1f}",
                        help="Kaptan olarak ortalama beklenen puan"
                    )
                    st.metric(
                        f"{player2_name} Ortalama",
                        f"{duel_analysis[player2_name]['mean_captaincy_points']:.1f}",
                        help="Kaptan olarak ortalama beklenen puan"
                    )

                with stats_col2:
                    p1_win_rate = duel_analysis['comparison'][f'{player1_name}_win_rate']
                    p2_win_rate = duel_analysis['comparison'][f'{player2_name}_win_rate']

                    st.metric(
                        f"{player1_name} Kazanma Oranı",
                        f"{p1_win_rate:.1%}",
                        help="Kaptan olarak diğerini geçme ihtimali"
                    )
                    st.metric(
                        f"{player2_name} Kazanma Oranı",
                        f"{p2_win_rate:.1%}",
                        help="Kaptan olarak diğerini geçme ihtimali"
                    )

                with stats_col3:
                    p1_mega = duel_analysis[player1_name]['prob_mega_haul']
                    p2_mega = duel_analysis[player2_name]['prob_mega_haul']

                    st.metric(
                        f"{player1_name} Mega Haul",
                        f"{p1_mega:.1%}",
                        help=">15 puan alma ihtimali (kaptan olarak)"
                    )
                    st.metric(
                        f"{player2_name} Mega Haul",
                        f"{p2_mega:.1%}",
                        help=">15 puan alma ihtimali (kaptan olarak)"
                    )

                # Decision recommendation
                st.markdown("---")
                st.markdown("### 💡 Kaptanlık Önerisi")

                p1_mean = duel_analysis[player1_name]['mean_captaincy_points']
                p2_mean = duel_analysis[player2_name]['mean_captaincy_points']
                p1_volatility = duel_analysis['comparison']['p1_volatility']
                p2_volatility = duel_analysis['comparison']['p2_volatility']

                if p1_mean > p2_mean:
                    winner = player1_name
                    loser = player2_name
                    winner_volatility = p1_volatility
                    loser_volatility = p2_volatility
                else:
                    winner = player2_name
                    loser = player1_name
                    winner_volatility = p2_volatility
                    loser_volatility = p1_volatility

                recommendation = f"""
                **Önerilen Kaptan: {winner}**

                **Neden?**
                - {winner} ortalama {abs(p1_mean - p2_mean):.1f} puan daha yüksek performans bekliyor
                - {winner} volatilitesi: {winner_volatility:.1f}, {loser} volatilitesi: {loser_volatility:.1f}
                """

                if winner_volatility > loser_volatility:
                    recommendation += f"- **Risk-Sevap Dengesi:** {winner}'ın patlama ihtimali daha yüksek ama daha tutarsız"
                else:
                    recommendation += f"- **Güvenilir Seçim:** {winner} daha tutarlı performans sunuyor"

                st.success(recommendation)

        elif player1_name == player2_name and player1_name != "Seçiniz...":
            st.warning("Lütfen farklı iki oyuncu seçin!")

        else:
            st.info("Kaptanlık düellosu için iki farklı oyuncu seçin.")


if __name__ == "__main__":
    main()


