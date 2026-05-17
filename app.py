import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ==================== НАСТРОЙКИ ====================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI HR Скринер", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AI HR Скринер")
st.header("Подбор кандидатов под вакансию")

st.divider()

# ==================== ИНИЦИАЛИЗАЦИЯ SESSION_STATE ====================
if "candidates_all" not in st.session_state:
    st.session_state.candidates_all = []  # Все кандидаты из последнего поиска
if "rejected_ids" not in st.session_state:
    st.session_state.rejected_ids = set()  # ID отсеянных кандидатов
if "last_search_time" not in st.session_state:
    st.session_state.last_search_time = None

# ==================== НАСТРОЙКИ ФИЛЬТРАЦИИ ====================
with st.sidebar:
    st.header("Настройки фильтрации")
    
    st.subheader("Возраст")
    min_age = st.number_input("От", min_value=14, max_value=100, value=None, placeholder="Не указано", label_visibility="collapsed")
    max_age = st.number_input("До", min_value=14, max_value=100, value=None, placeholder="Не указано", label_visibility="collapsed")
    
    st.subheader("Опыт работы")
    min_exp = st.number_input("Мин. опыт (лет)", min_value=0, max_value=50, value=None, placeholder="Не указано")
    
    st.subheader("Город")
    city_input = st.text_input("Город", value="", placeholder="Оставьте пустым для отключения")
    
    st.subheader("Ключевые слова в должности")
    pos_kw_input = st.text_input("Через запятую", value="", placeholder="Backend, Python, Data Scientist")
    pos_kw = [kw.strip() for kw in pos_kw_input.split(",") if kw.strip()]
    
    st.subheader("Ключевые навыки")
    skill_kw_input = st.text_input("Через запятую", value="", placeholder="Python, SQL, Docker, FastAPI")
    skill_kw = [kw.strip() for kw in skill_kw_input.split(",") if kw.strip()]
    
    st.divider()
    
    st.subheader("Параметры поиска")
    min_score = st.slider("Мин. процент совпадения", min_value=0.0, max_value=100.0, value=15.0, step=5.0)
    top_k = st.slider("Топ результатов", min_value=5, max_value=50, value=20, step=5)
    
    st.divider()
    
    # Кнопка сброса отсеянных
    if st.button("🔄 Сбросить всех отсеянных", use_container_width=True):
        st.session_state.rejected_ids.clear()
        st.rerun()

# ==================== ОСНОВНАЯ ОБЛАСТЬ ====================

st.subheader("Текст вакансии")
vacancy_text = st.text_area(
    "Опишите требования к кандидату",
    value="",
    height=200
)

st.divider()

uploaded = st.file_uploader("Загрузите файл с резюме", type="txt")

st.divider()

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_button = st.button("Запустить подбор", type="primary", use_container_width=True)

if run_button:
    if not uploaded:
        st.error("Ошибка: файл не загружен")
        st.stop()
    
    if not vacancy_text.strip():
        st.error("Ошибка: текст не введен")
        st.stop()
    
    files = {
        "vacancy_file": ("vacancy.txt", vacancy_text.encode("utf-8"), "text/plain"),
        "resume_file": ("resumes.txt", uploaded.getvalue(), "text/plain")
    }
    
    data = {
        "top_k": str(top_k),
        "min_score": str(min_score)
    }
    
    if min_age is not None:
        data["min_age"] = str(min_age)
    if max_age is not None:
        data["max_age"] = str(max_age)
    if min_exp is not None:
        data["min_experience"] = str(min_exp)
    if city_input.strip():
        data["city"] = city_input.strip()
    if pos_kw:
        data["position_keywords"] = ",".join(pos_kw)
    if skill_kw:
        data["skills_keywords"] = ",".join(skill_kw)
    
    with st.spinner("Анализ (15-40 секунд)..."):
        try:
            response = requests.post(
                f"{API_URL}/api/v1/match/advanced",
                files=files,
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("top_candidates", [])
                
                # Сохраняем в session_state с уникальными ID
                st.session_state.candidates_all = []
                for idx, c in enumerate(candidates):
                    # Генерируем уникальный ID для кандидата (на основе имени + города + возраста)
                    unique_id = f"{c.get('name', '')}_{c.get('city', '')}_{c.get('age', '')}_{idx}"
                    st.session_state.candidates_all.append({
                        "id": unique_id,
                        "data": c
                    })
                
                st.session_state.last_search_time = datetime.now()
                st.rerun()
            else:
                st.error(f"Ошибка API: {response.status_code}")
                st.code(response.text)
                
        except Exception as e:
            st.error(f"Ошибка: {e}")

# ==================== ОТОБРАЖЕНИЕ КАНДИДАТОВ ====================

if st.session_state.candidates_all:
    # Разделяем на активных и отсеянных
    active_candidates = [c for c in st.session_state.candidates_all 
                         if c["id"] not in st.session_state.rejected_ids]
    rejected_candidates = [c for c in st.session_state.candidates_all 
                           if c["id"] in st.session_state.rejected_ids]
    
    # Создаем вкладки
    tab_all, tab_rejected = st.tabs([
        f"✅ Активные кандидаты ({len(active_candidates)})",
        f"❌ Отсеянные ({len(rejected_candidates)})"
    ])
    
    # ========== ВКЛАДКА "АКТИВНЫЕ" ==========
    with tab_all:
        if active_candidates:
            # Таблица с активными
            df_data = []
            for c in active_candidates:
                cand = c["data"]
                row = {
                    "Имя": cand.get("name", "—"),
                    "Возраст": cand.get("age", "—"),
                    "Город": cand.get("city", "—"),
                    "Опыт": cand.get("experience", "—"),
                    "Совпадение %": cand.get("match_score", 0),
                    "Навыки": cand.get("skills", "—")[:50]
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            st.subheader("Детальная информация")
            
            for c in active_candidates:
                cand = c["data"]
                candidate_id = c["id"]
                name = cand.get("name", "Unknown")
                title = f"{name} - Совпадение: {cand.get('match_score', 0):.1f}%"
                
                with st.expander(title):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**Имя:** {cand.get('name', '—')}")
                        st.write(f"**Возраст:** {cand.get('age', '—')}")
                        st.write(f"**Опыт:** {cand.get('experience', '—')} лет")
                    with col2:
                        st.write(f"**Город:** {cand.get('city', '—')}")
                        st.write(f"**Зарплата:** {cand.get('salary', '—')}")
                        st.write(f"**Совпадение:** {cand.get('match_score', 0):.1f}%")
                    with col3:
                        st.write("")
                        st.write("")
                        if st.button("❌ Отсеять", key=f"reject_{candidate_id}"):
                            st.session_state.rejected_ids.add(candidate_id)
                            st.rerun()
                    
                    if cand.get('skills'):
                        skills_str = cand.get('skills', '—')
                        if len(skills_str) > 300:
                            skills_str = skills_str[:300] + "..."
                        st.write(f"**Навыки:** {skills_str}")
                    
                    if cand.get('desired_position'):
                        st.write(f"**Желаемая должность:** {cand.get('desired_position', '—')}")
        else:
            st.info("Нет активных кандидатов")
    
    # ========== ВКЛАДКА "ОТСЕЯННЫЕ" ==========
    with tab_rejected:
        if rejected_candidates:
            # Таблица с отсеянными
            df_rejected = []
            for c in rejected_candidates:
                cand = c["data"]
                row = {
                    "Имя": cand.get("name", "—"),
                    "Возраст": cand.get("age", "—"),
                    "Город": cand.get("city", "—"),
                    "Причина": "Отсеян пользователем",
                    "Совпадение %": cand.get("match_score", 0)
                }
                df_rejected.append(row)
            
            df_r = pd.DataFrame(df_rejected)
            st.dataframe(df_r, use_container_width=True, hide_index=True)
            
            st.subheader("Восстановить кандидатов")
            for c in rejected_candidates:
                cand = c["data"]
                candidate_id = c["id"]
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{cand.get('name', 'Unknown')}** — {cand.get('match_score', 0):.1f}%")
                with col2:
                    if st.button("↩️ Вернуть", key=f"restore_{candidate_id}"):
                        st.session_state.rejected_ids.discard(candidate_id)
                        st.rerun()
        else:
            st.info("Нет отсеянных кандидатов")
    
    # Информация о времени последнего поиска
    if st.session_state.last_search_time:
        st.caption(f"📅 Последний поиск: {st.session_state.last_search_time.strftime('%H:%M:%S')}")

st.divider()
st.caption("AI HR Скринер — статусы хранятся только в текущей сессии")
