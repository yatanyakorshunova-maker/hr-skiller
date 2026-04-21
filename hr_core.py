import torch
import re
import warnings
import numpy as np
from scipy.special import softmax
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder

warnings.filterwarnings('ignore')

# Кастомные библиотеки
from x_russian_cities import is_valid_russian_city, normalize_city


# ====================== ПЕРЕЗАГРУЗКА МОДУЛЯ ======================
import importlib
import x_it_keywords

importlib.reload(x_it_keywords)
from x_it_keywords import is_it_candidate

print(f"Количество IT-ключевых слов: {x_it_keywords.get_it_keywords_count()}")


# Загрузка моделей
model = SentenceTransformer("ai-forever/sbert_large_nlu_ru")
reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    max_length=512,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("Модели успешно загружены")


def extract_filters_from_vacancy(vacancy_text: str) -> dict:
    """Автоматически извлекает фильтры из текста вакансии"""
    text = vacancy_text.lower()
    filters = {
        'min_age': None,
        'max_age': None,
        'min_experience': None,
        'city': None,
        'position_keywords': [],
        'skills_keywords': []
    }

    # Возраст
    age_matches = re.findall(r'возраст[а-я\\s]*(\\d{1,2})\\s*[-–—]\\s*(\\d{1,2})', text)
    if age_matches:
        filters['min_age'] = int(age_matches[0][0])
        filters['max_age'] = int(age_matches[0][1])
    else:
        min_age_match = re.search(r'(?:возраст|от)\\s*(\\d{1,2})\\s*(?:год|лет)', text)
        if min_age_match:
            filters['min_age'] = int(min_age_match.group(1))

    # Опыт работы
    exp_matches = re.findall(r'(?:опыт|стаж)[а-я\\s]*(\\d{1,2})\\s*(?:год|лет|года)', text)
    if exp_matches:
        filters['min_experience'] = int(exp_matches[0])

    # Город
    city_match = re.search(r'(?:в|город|локация|регион)[:\\s]+([А-Яа-яЁё\\s-]+)', text)
    if city_match:
        city_name = city_match.group(1).strip()
        if is_valid_russian_city(city_name):
            filters['city'] = city_name

    # Ключевые слова должности (простые популярные варианты)
    position_patterns = ['data scientist', 'ml engineer', 'machine learning', 'backend', 'frontend', 
                         'fullstack', 'python developer', 'java developer', 'middle', 'senior']
    for pat in position_patterns:
        if pat in text:
            filters['position_keywords'].append(pat.title())

    # Навыки (самые частые)
    skill_patterns = ['python', 'fastapi', 'django', 'flask', 'sql', 'postgresql', 'docker', 
                      'kubernetes', 'redis', 'pytorch', 'tensorflow', 'pandas', 'react', 'typescript']
    for skill in skill_patterns:
        if skill in text:
            filters['skills_keywords'].append(skill.capitalize())

    # Убираем дубликаты
    filters['position_keywords'] = list(dict.fromkeys(filters['position_keywords']))
    filters['skills_keywords'] = list(dict.fromkeys(filters['skills_keywords']))

    return filters


def build_resume_text(resume: Dict) -> str:
    parts = []
    if name := resume.get('name'):
        parts.append(f"Имя: {name.strip()}")
    age = resume.get('parsed_age') or resume.get('age')
    if age:
        parts.append(f"Возраст: {age}")
    if city := resume.get('city'):
        parts.append(f"Город: {city.strip()}")
    if position := resume.get('desired_position'):
        parts.append(f"Желаемая должность: {position.strip()}")
    exp = resume.get('parsed_experience') or resume.get('experience')
    if exp is not None:
        parts.append(f"Опыт работы: {exp} лет")
    if skills := resume.get('skills'):
        parts.append(f"Навыки: {skills.strip()}")
    if education := resume.get('education'):
        parts.append(f"Образование: {education.strip()}")
    if last_job := resume.get('last_job'):
        parts.append(f"Последнее место работы: {last_job.strip()}")
    if comment := resume.get('comment'):
        parts.append(f"Комментарий: {comment.strip()}")
    return "\n".join(parts).strip()


def extract_age(text: str) -> Optional[int]:
    if not text:
        return None
    text_str = str(text).strip().lower()
    if text_str in ['неизвестен', 'unknown', '-', '', 'нет', ' ']:
        return None
    text_str = re.sub(r'\s*(год|лет|года|лет)[а-я]*.*$', '', text_str, flags=re.IGNORECASE)
    matches = re.findall(r'\b(\d{1,2})\b', text_str)
    for m in matches:
        age = int(m)
        if 16 <= age <= 65:
            return age
    for m in matches:
        age = int(m)
        if 15 <= age <= 70:
            return age
    return None


def extract_experience(text: str) -> Optional[int]:
    if not text:
        return None
    text_str = str(text).strip().lower()
    if text_str in ['неизвестен', 'unknown', 'нет', '∞', '-', '', '0 (minecraft)']:
        return None
    text_str = re.sub(r'\+7\s*\(\d{3}\)\s*\d{3}-\d{2}-\d{2}', '', text_str)
    text_str = re.sub(r'\s*(год|лет|года)[а-я]*.*$', '', text_str, flags=re.IGNORECASE)
    matches = re.findall(r'\b(\d{1,2})\b', text_str)
    for m in matches:
        exp = int(m)
        if 0 <= exp <= 50:
            return exp
    return None


def is_valid_name(name: str) -> bool:
    if not name:
        return False
    name = name.strip()
    if len(name) < 5:
        return False
    name = re.sub(r'\s+', ' ', name)
    forbidden = {
        'test@mail.ru', 'неизвестный', 'неизвестен', 'unknown', 'anonymous',
        'java', 'python', 'spring', 'docker', 'sql', 'html', 'css'
    }
    lower_name = name.lower()
    if any(word in lower_name for word in forbidden):
        return False
    if re.search(r'\d|@|\.ru|\.com', name):
        return False
    words = name.split()
    if len(words) < 2 or len(words) > 4:
        return False
    for word in words:
        if not re.match(r'^[А-ЯЁ]', word) or not re.match(r'^[А-Яа-яЁё-]+$', word) or len(word) < 2:
            return False
    if len(words) == 1 and len(name) > 12:
        return False
    single_word_lower = words[0].lower()
    if len(words) == 1 and len(single_word_lower) > 8 and re.match(r'^[а-яё]+$', single_word_lower):
        return False
    return True


def load_resumes_from_file(filename: str) -> List[Dict]:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'(=== Резюме кандидата №\d+ ===)', content)
    resumes = []
    resume_number = 0

    for i in range(1, len(blocks), 2):
        header = blocks[i].strip()
        block = blocks[i+1] if i+1 < len(blocks) else ""

        match = re.search(r'№(\d+)', header)
        resume_number = int(match.group(1)) if match else resume_number + 1

        resume = {
            'resume_number': resume_number,
            'name': '', 'age': None, 'experience': None, 'city': '',
            'desired_position': '', 'skills': '', 'education': '',
            'salary': '', 'last_job': '', 'comment': ''
        }

        lines = [line.strip() for line in block.split('\n') if line.strip()]

        for line in lines:
            if line.startswith('Имя:'):
                resume['name'] = line[4:].strip()
            elif line.startswith('Возраст:'):
                resume['age'] = line[8:].strip()
            elif line.startswith('Город:'):
                resume['city'] = line[6:].strip()
            elif line.startswith('Желаемая должность:'):
                resume['desired_position'] = line[19:].strip()
            elif line.startswith('Опыт') and ':' in line:
                resume['experience'] = line.split(':', 1)[1].strip()
            elif line.startswith('Навыки:'):
                resume['skills'] = line[7:].strip()
            elif line.startswith('Образование:'):
                resume['education'] = line[12:].strip()
            elif line.startswith('Ожидаемая зарплата:'):
                resume['salary'] = line[19:].strip()
            elif line.startswith('Последнее место работы:'):
                resume['last_job'] = line[24:].strip()
            elif line.startswith('Комментарий:'):
                resume['comment'] = line[12:].strip()

        if not resume['name'] or re.match(r'^\d{1,2}\s*лет?$', resume['name'].strip()):
            resume['name'] = "Неизвестный кандидат"

        # Попытка вытащить возраст и опыт из других полей
        if not resume.get('age'):
            for field in ['name', 'desired_position', 'city', 'comment', 'experience']:
                if resume.get(field):
                    extracted = extract_age(resume[field])
                    if extracted:
                        resume['age'] = str(extracted)
                        break

        if not resume.get('experience'):
            for field in ['experience', 'name', 'comment']:
                if resume.get(field):
                    extracted = extract_experience(resume[field])
                    if extracted is not None:
                        resume['experience'] = str(extracted)
                        break

        resumes.append(resume)

    print(f"Загружено {len(resumes)} резюме из файла {filename}")
    return resumes


def rerank_candidates(vacancy_text: str, candidates: List[Dict], top_k: int = 20) -> List[Dict]:
    if not candidates:
        return []
    
    pairs = []
    for cand in candidates:
        resume_dict = {
            'name': cand.get('name', ''),
            'parsed_age': cand.get('age'),
            'parsed_experience': cand.get('experience'),
            'city': cand.get('city', ''),
            'desired_position': cand.get('desired_position', ''),
            'skills': cand.get('skills', ''),
            'education': cand.get('education', ''),
            'last_job': cand.get('last_job', ''),
            'comment': cand.get('comment', '')
        }
        pairs.append([vacancy_text, build_resume_text(resume_dict)])

    raw_scores = reranker.predict(pairs, show_progress_bar=False)
    scores_array = np.array(raw_scores)
    normalized = softmax(scores_array) * 100

    for i, cand in enumerate(candidates):
        cand['rerank_score'] = float(raw_scores[i])
        cand['rerank_score_percent'] = round(float(normalized[i]), 1)

    return sorted(candidates, key=lambda x: x.get('rerank_score_percent', 0), reverse=True)[:top_k]


def rank_candidates(
    vacancy_text: str,
    resumes: List[Dict],
    manual_filters: dict = None,
    use_filters: bool = True,
    use_reranking: bool = True,
    min_score: float = 15.0,
    top_k: int = 20,
    rerank_top: int = 40,
    it_threshold: float = 6.0          # Порог для IT-фильтра
) -> List[Dict]:
    
    # ====================== ФОРМИРОВАНИЕ ФИЛЬТРОВ ======================
    filters = {}

    # Автоматический фильтр
    if USE_AUTO_FILTERS:
        print("Автоматическое извлечение фильтров из текста вакансии...")
        auto_filters = extract_filters_from_vacancy(vacancy_text)
        filters.update(auto_filters)

    # Ручной фильтр (переопределяет автоматический)
    if USE_MANUAL_FILTERS and manual_filters and len(manual_filters) > 0:
        print("Применяются ручные фильтры (переопределяют автоматические)")
        filters.update(manual_filters)

    if not filters:
        print("Фильтры отключены (ни автоматический, ни ручной не активны)")
    else:
        print(f"Применяемые фильтры: {filters}\n")

    filtered_resumes = []
    near_miss_resumes = []
    removed_stats = {
        "invalid_name": 0, "invalid_city": 0, "no_age": 0, "no_experience": 0,
        "non_it": 0, "filter_age": 0, "filter_experience": 0, "filter_city": 0,
        "filter_position": 0, "filter_skills": 0, "low_score": 0
    }

    print("Начинаем валидацию и фильтрацию резюме...\n")

    for resume in resumes:
        reason = None
        raw_name = str(resume.get('name', '')).strip()
        raw_city = str(resume.get('city', '')).strip()
        raw_position = str(resume.get('desired_position', '')).lower()
        raw_skills = str(resume.get('skills', '')).lower()

        age = extract_age(resume.get('age')) if resume.get('age') else None
        experience = extract_experience(resume.get('experience')) if resume.get('experience') else None

        # ====================== ЖЁСТКАЯ ВАЛИДАЦИЯ ======================
        if not raw_name or not is_valid_name(raw_name):
            removed_stats["invalid_name"] += 1
            reason = f"Некорректное имя: '{raw_name}'"
        
        elif age is None or not (16 <= age <= 70):
            removed_stats["no_age"] += 1
            reason = f"Некорректный возраст (было: {resume.get('age')})"
        
        elif experience is None or not (0 <= experience <= 50):
            removed_stats["no_experience"] += 1
            reason = f"Некорректный опыт (было: {resume.get('experience')})"
        
        elif not is_valid_russian_city(raw_city):
            removed_stats["invalid_city"] += 1
            reason = f"Неизвестный или некорректный город: '{raw_city}'"
        
        elif len(raw_name.split()) <= 1 and is_valid_russian_city(raw_name):
            removed_stats["invalid_name"] += 1
            reason = "Имя и город, скорее всего, перепутаны местами"

        # ====================== ПРОВЕРКА НА IT-КАНДИДАТА ======================
        elif True:  # всегда выполняем проверку
            is_it, it_reason = is_it_candidate(resume, threshold=it_threshold)
            if not is_it:
                removed_stats["non_it"] += 1
                reason = f"Не подходящая IT-специальность ({it_reason})"

        # ====================== HR-ФИЛЬТРЫ ======================
        hr_filter_reason = None
        if reason is None and use_filters and filters:
            if USE_MIN_AGE and filters.get('min_age') and age is not None and age < filters['min_age']:
                hr_filter_reason = f"Возраст {age} ниже минимального"
                removed_stats["filter_age"] += 1
            elif USE_MAX_AGE and filters.get('max_age') and age is not None and age > filters['max_age']:
                hr_filter_reason = f"Возраст {age} выше максимального"
                removed_stats["filter_age"] += 1
            elif USE_MIN_EXPERIENCE and filters.get('min_experience') and experience is not None and experience < filters['min_experience']:
                hr_filter_reason = f"Опыт {experience} лет меньше требуемого"
                removed_stats["filter_experience"] += 1
            elif USE_CITY and filters.get('city') and filters['city'].lower() not in normalize_city(raw_city):
                hr_filter_reason = f"Город не соответствует фильтру ({filters['city']})"
                removed_stats["filter_city"] += 1
            elif USE_POSITION_KEYWORDS and filters.get('position_keywords'):
                pos_match = any(kw.lower() in raw_position for kw in filters['position_keywords'])
                if not pos_match:
                    hr_filter_reason = f"Должность не подходит ({resume.get('desired_position', '—')})"
                    removed_stats["filter_position"] += 1
            elif USE_SKILLS_KEYWORDS and filters.get('skills_keywords'):
                skill_match = any(kw.lower() in raw_skills for kw in filters['skills_keywords'])
                if not skill_match:
                    hr_filter_reason = f"Навыки не содержат ключевых слов"
                    removed_stats["filter_skills"] += 1

        # ====================== СОХРАНЕНИЕ РЕЗЮМЕ ======================
        if reason is None:
            resume['parsed_age'] = age
            resume['parsed_experience'] = experience

            if hr_filter_reason is None:
                filtered_resumes.append(resume)
            else:
                resume['hr_filter_reason'] = hr_filter_reason
                near_miss_resumes.append(resume)
        else:
            print(f"Отсеяно | {raw_name[:35]:35} | Город: {raw_city[:25]:25} | Причина: {reason}")

    # ====================== ОТЧЁТ ======================
    total_removed = sum(removed_stats.values())
    print("\n" + "="*100)
    print(f"Результат фильтрации: {len(filtered_resumes)} полностью подходящих резюме из {len(resumes)}")
    print(f"Отсеяно: {total_removed} резюме")
    
    if total_removed > 0:
        print("\nПричины отсева:")
        reason_map = {
            "invalid_name": "Некорректное имя",
            "invalid_city": "Неверный / неизвестный город",
            "no_age": "Проблема с возрастом",
            "no_experience": "Проблема с опытом",
            "non_it": "Не подходящая IT-специальность",
            "filter_age": "Не прошёл фильтр по возрасту",
            "filter_experience": "Не прошёл фильтр по опыту",
            "filter_city": "Не прошёл фильтр по городу",
            "filter_position": "Должность не подходит",
            "filter_skills": "Не прошёл фильтр по навыкам",
            "low_score": "Низкий семантический score"
        }
        for key, count in removed_stats.items():
            if count > 0:
                print(f"   • {reason_map.get(key, key)}: {count}")

    print("="*100)

    if not filtered_resumes and not near_miss_resumes:
        print("Кандидатов не найдено.")
        return []

    # ====================== СЕМАНТИЧЕСКОЕ РАНЖИРОВАНИЕ ======================
    print(f"\nВыполняется семантическое ранжирование...")

    vacancy_embedding = model.encode(vacancy_text, convert_to_tensor=True)

    results = []
    for resume in filtered_resumes:
        resume_text = build_resume_text(resume)
        if len(resume_text) < 30: 
            continue
        similarity = torch.nn.functional.cosine_similarity(
            vacancy_embedding.unsqueeze(0),
            model.encode(resume_text, convert_to_tensor=True).unsqueeze(0)
        ).item()
        score = round(similarity * 100, 1)
        if score < min_score: 
            continue
        results.append({**resume, 'score': score})

    results.sort(key=lambda x: x['score'], reverse=True)

    if use_reranking and len(results) > 3:
        print(f"Запускаем reranking топ-{min(rerank_top, len(results))} кандидатов...")
        results = rerank_candidates(vacancy_text, results[:rerank_top], top_k)

    final_results = results[:top_k]

    # ====================== ВЫВОД БЛИЗКИХ КАНДИДАТОВ ======================
    if near_miss_resumes:
        print("\n" + "="*130)
        print("БЛИЗКИЕ КАНДИДАТЫ (IT-специалисты, но не прошли HR-фильтры):")
        print("="*130)
        
        near_results = []
        for resume in near_miss_resumes:
            resume_text = build_resume_text(resume)
            if len(resume_text) < 30: 
                continue
            similarity = torch.nn.functional.cosine_similarity(
                vacancy_embedding.unsqueeze(0),
                model.encode(resume_text, convert_to_tensor=True).unsqueeze(0)
            ).item()
            score = round(similarity * 100, 1)
            near_results.append({**resume, 'score': score})

        near_results.sort(key=lambda x: x['score'], reverse=True)

        for i, cand in enumerate(near_results[:15], 1):
            print(f"{i:2d}. {cand['name']:38} | Возраст: {cand.get('parsed_age') or cand.get('age'):3} | "
                  f"Опыт: {cand.get('parsed_experience') or cand.get('experience'):2} лет | "
                  f"Город: {cand.get('city', '-'):15} | Score: {cand['score']:5.1f}% | "
                  f"Причина: {cand.get('hr_filter_reason', '—')} (№{cand.get('resume_number')})")

    # ====================== ВЫВОД ТОП КАНДИДАТОВ ======================
    print("\n" + "="*130)
    print(f"ТОП-{top_k} КАНДИДАТОВ (полностью соответствуют всем фильтрам) | Reranking: {'ВКЛ' if use_reranking else 'ВЫКЛ'}")
    print("="*130)

    for i, cand in enumerate(final_results, 1):
        rerank = cand.get('rerank_score_percent')
        score_str = f"Retrieval: {cand['score']:5.1f}% → Rerank: {rerank:5.1f}%" if rerank is not None else f"Retrieval: {cand['score']:5.1f}%"
        print(f"{i:2d}. {cand['name']:38} | Возраст: {cand.get('parsed_age') or cand.get('age'):3} | "
              f"Опыт: {cand.get('parsed_experience') or cand.get('experience'):2} лет | "
              f"Город: {cand.get('city', '-'):15} | З/п: {cand.get('salary', '-'):8} | {score_str} (№{cand.get('resume_number')})")

    return final_results


# ====================== НАСТРОЙКИ ДЛЯ HR ======================

vacancy_text = """
Ищем Middle Backend Developer (Python) в развивающийся стартап.

Требования:
- Опыт коммерческой разработки на Python от 4 лет
- Уверенное владение FastAPI или Django
- Опыт работы с PostgreSQL и SQL
- Знание Docker и основ Kubernetes
- Опыт работы с Redis и очередями
- Возраст от 23 до 45 лет
- Город: любой (готовы рассматривать релокацию)

Будет плюсом: опыт с GitLab CI/CD, AWS, микросервисами.
"""

# ====================== РУЧНЫЕ ФИЛЬТРЫ ======================
manual_filters = {
    'min_age': 23,
    'max_age': 45,
    'min_experience': 5,
    'city': None,
    'position_keywords': ["Backend", "Python", "Developer"],
    'skills_keywords': ["Python", "FastAPI", "PostgreSQL", "Docker"]
}

# ====================== ВКЛ/ВЫКЛ ФИЛЬТРОВ ======================
USE_AUTO_FILTERS = True
USE_MANUAL_FILTERS = True

USE_MIN_AGE = True
USE_MAX_AGE = True
USE_MIN_EXPERIENCE = True
USE_CITY = False
USE_POSITION_KEYWORDS = True
USE_SKILLS_KEYWORDS = True

USE_FILTERS = True
USE_RERANKING = True
MIN_SCORE = 15.0
TOP_K = 20

# Порог чувствительности IT-фильтра
IT_THRESHOLD = 2.0

# ====================== ЗАПУСК ======================
print("Запуск системы подбора кандидатов\n")

resumes = load_resumes_from_file("resumes_generated.txt")

top_candidates = rank_candidates(
    vacancy_text=vacancy_text,
    resumes=resumes,
    manual_filters=manual_filters,
    use_filters=USE_FILTERS,
    use_reranking=USE_RERANKING,
    min_score=MIN_SCORE,
    top_k=TOP_K,
    it_threshold=IT_THRESHOLD
)

# ====================== ИТОГОВЫЙ ОТЧЁТ ======================
print("\n" + "="*130)
print(f"ГОТОВО | Найдено {len(top_candidates)} лучших кандидатов")
print("="*130)
print(f"Вакансия: Middle Backend Developer (Python)")
print(f"Фильтры: Авто={USE_AUTO_FILTERS}, Ручные={USE_MANUAL_FILTERS} | IT Threshold: {IT_THRESHOLD}")
print("="*130)