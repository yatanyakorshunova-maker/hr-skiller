import streamlit as st
import requests
import pandas as pd

# ==================== НАСТРОЙКИ ====================
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI HR Скринер", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== ЗАГОЛОВОК ====================
st.title("AI HR Скринер")
st.header("Подбор кандидатов под вакансию")

st.divider()

# ==================== НАСТРОЙКИ ФИЛЬТРАЦИИ ====================
with st.sidebar:
    st.header("Настройки фильтрации")
    
    st.subheader("Возраст")
    col1, col2 = st.columns(2)
    with col1:
        min_age = st.number_input("От", min_value=14, max_value=100, value=None, placeholder="Не указано")
    with col2:
        max_age = st.number_input("До", min_value=14, max_value=100, value=None, placeholder="Не указано")
    
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

# ==================== ОСНОВНАЯ ОБЛАСТЬ ====================

st.subheader("Текст вакансии")
vacancy_text = st.text_area(
    "Опишите требования к кандидату",
    value="",
    height=200,
    placeholder="""
Пример:
Ищем Python разработчика (Middle/Senior). Опыт от 3 лет.
Обязательные навыки: FastAPI, PostgreSQL, Docker.
Возраст: 25-40 лет.
Город: Москва или удаленно.
    """
)

st.divider()

uploaded = st.file_uploader("Загрузите файл с резюме (resumes_generated.txt)", type="txt")

st.divider()

# ==================== КНОПКА ЗАПУСКА ====================
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
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("top_candidates", [])
                
                if candidates:
                    st.success(f"Найдено {len(candidates)} подходящих кандидатов")
                    
                    df_data = []
                    for c in candidates:
                        row = {
                            "Имя": c.get("name", "—"),
                            "Возраст": c.get("age", "—"),
                            "Город": c.get("city", "—"),
                            "Опыт": c.get("experience", "—"),
                            "Совпадение %": c.get("match_score", 0),
                            "Навыки": c.get("skills", "—")[:50]
                        }
                        df_data.append(row)
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.subheader("Детальная информация")
                    
                    for idx, c in enumerate(candidates):
                        name = c.get("name", "Unknown")
                        title = f"{idx+1}. {name} - Совпадение: {c.get('match_score', 0):.1f}%"
                        
                        with st.expander(title):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Имя:** {c.get('name', '—')}")
                                st.write(f"**Возраст:** {c.get('age', '—')}")
                                st.write(f"**Опыт:** {c.get('experience', '—')} лет")
                                if c.get('education'):
                                    st.write(f"**Образование:** {c.get('education', '—')[:200]}")
                            with col2:
                                st.write(f"**Город:** {c.get('city', '—')}")
                                st.write(f"**Зарплата:** {c.get('salary', '—')}")
                                st.write(f"**Должность:** {c.get('desired_position', '—')}")
                                st.write(f"**Совпадение:** {c.get('match_score', 0):.1f}%")
                                if c.get('rerank_score_percent'):
                                    st.write(f"**Точное совпадение:** {c.get('rerank_score_percent', 0):.1f}%")
                            
                            if c.get('skills'):
                                skills_str = c.get('skills', '—')
                                if len(skills_str) > 300:
                                    skills_str = skills_str[:300]
                                st.write(f"**Навыки:** {skills_str}")
                            
                            if c.get('last_job'):
                                st.write(f"**Последнее место работы:** {c.get('last_job', '—')}")
                else:
                    st.warning("Ничего не найдено. Попробуйте снизить порог совпадения.")
            else:
                st.error(f"Ошибка API: {response.status_code}")
                st.code(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error(f"Не удалось подключиться к бэкенду. Убедитесь, что он запущен на {API_URL}")
        except requests.exceptions.Timeout:
            st.error("Превышено время ожидания. Попробуйте уменьшить количество резюме.")
        except Exception as e:
            st.error(f"Ошибка: {e}")

st.divider()
st.caption("AI HR Скринер | Работает через бэкенд FastAPI")
