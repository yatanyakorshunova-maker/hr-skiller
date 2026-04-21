"""
x_it_keywords.py — Улучшенная проверка IT-кандидатов с понятными причинами отсева
"""

IT_KEYWORDS_WITH_WEIGHTS = {
    "developer": 5, "разработчик": 5, "программист": 5, "engineer": 5, "инженер": 5,
    "android developer": 5, "ios developer": 5, "mobile developer": 5,
    "мобильный разработчик": 5, "андроид разработчик": 5,
    "data scientist": 5, "data analyst": 5, "data engineer": 5,
    "ml engineer": 5, "machine learning": 5, "ai engineer": 5,
    "frontend": 5, "backend": 5, "fullstack": 5, "фуллстек": 5,
    "devops": 5, "sre": 5, "qa": 5, "тестировщик": 5, "sdet": 5,
    "product manager": 5, "продакт": 5, "product owner": 5,
}

GOOD_IT_POSITIONS = {
    "system analyst", "системный аналитик", "business analyst", "бизнес-аналитик",
    "security analyst", "инженер по информационной безопасности",
    "product manager", "продакт менеджер", "project manager", "тимлид", "tech lead",
    "scrum master", "архитектор"
}

NON_IT_POSITIONS = {
    'водитель', 'парикмахер', 'бухгалтер', 'учитель', 'врач', 'медсестра',
    'сантехник', 'повар', 'кассир', 'продавец', 'охранник', 'грузчик', 'курьер',
    'менеджер по продажам', 'hr', 'рекрутер', 'логист', 'экономист', 'юрист',
    'sales manager', 'hr generalist'
}


def is_it_candidate(resume: dict, threshold: float = 6.0) -> tuple:
    """
    Возвращает (is_it: bool, reason: str)
    """
    if not resume:
        return False, "Пустое резюме"

    position = str(resume.get('desired_position', '')).lower().strip()
    skills = str(resume.get('skills', '')).lower().strip()
    last_job = str(resume.get('last_job', '')).lower().strip()
    comment = str(resume.get('comment', '')).lower().strip()

    full_text = f"{position} {skills} {last_job} {comment}".strip()

    # Жёсткая отсечка явных не-IT
    if any(non_it in position for non_it in NON_IT_POSITIONS):
        return False, f"Явно не-IT должность: {resume.get('desired_position', '—')}"

    # Хорошие IT-специальности (даже если мало ключевых слов)
    if any(good in position for good in GOOD_IT_POSITIONS):
        return True, "Подходящая IT-специальность"

    # Подсчёт баллов
    total_score = 0.0
    for keyword, weight in IT_KEYWORDS_WITH_WEIGHTS.items():
        if keyword in full_text:
            total_score += weight

    if any(word in position for word in ["senior", "middle", "lead", "тимлид", "tech lead"]):
        total_score += 2

    if any(tech in full_text for tech in ["python", "java", "kotlin", "android", "ios", "flutter", 
                                         "docker", "kubernetes", "fastapi", "sql"]):
        total_score += 3

    if total_score >= 10.0:
        return True, "Высокий IT-score"
    if total_score >= threshold:
        return True, f"Достаточный IT-score ({total_score:.1f})"

    return False, f"Недостаточно IT-признаков (score: {total_score:.1f})"


def get_it_keywords_count() -> int:
    return len(IT_KEYWORDS_WITH_WEIGHTS)