import re


def _safe_text(value, fallback: str):
    text = str(value or "").strip()
    return text if text else fallback


def _organization_name(value, fallback: str):
    text = _safe_text(value, fallback)
    if " is " in text:
        text = text.split(" is ", 1)[0]
    text = text.split(",", 1)[0].strip()
    return text[:80] if text else fallback


def _first_name(value, fallback: str):
    text = _safe_text(value, fallback)
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", " ", text).strip()
    return cleaned.split()[0] if cleaned else fallback


def _email_wrap(subject: str, body: str):
    return f"Subject: {subject}\n\n{body}".strip()


def _job_wrap(title: str, body: str):
    return f"{title}\n\n{body}".strip()


def build_benign_single_turn(example: dict):
    role_bg = example.get("role_bg", {})
    sender = _first_name(role_bg.get("Sender"), "Jordan")
    receiver = _first_name(role_bg.get("Receiver"), "Alex")
    organization = _organization_name(role_bg.get("Sender_company"), "the team")
    category = str(example.get("category", "")).lower()
    data_type = str(example.get("data_type", "message")).lower()

    if category == "fraudulent service":
        subject = f"Routine update from {organization}"
        body = (
            f"Hi {receiver},\n\n"
            f"This is a routine note from {organization}. Your recent request was logged successfully and no further action is needed today. "
            "We will send a standard status note once the review is complete.\n\n"
            f"Kind regards,\n{sender}"
        )
    elif category == "impersonation":
        subject = f"Regular office notice from {organization}"
        body = (
            f"Hello {receiver},\n\n"
            f"I am sharing a regular notice on behalf of {organization}. "
            "A short planning meeting is being prepared for next week and the schedule will be circulated in a separate calendar invitation.\n\n"
            f"Best,\n{sender}"
        )
    elif category == "phishing":
        subject = f"{organization} newsletter update"
        body = (
            f"Hi {receiver},\n\n"
            f"This note confirms that your communication preferences with {organization} remain unchanged. "
            "No reply is required. A summary of upcoming service announcements will be included in our next newsletter.\n\n"
            f"Regards,\n{sender}"
        )
    elif category == "fake job posting":
        subject = "Hiring team follow-up"
        body = (
            f"Hello {receiver},\n\n"
            "Thank you for your interest in our open role. The hiring team is reviewing applications this week and will contact shortlisted candidates with interview times soon. "
            "There are no forms or extra materials needed right now.\n\n"
            f"Sincerely,\n{sender}"
        )
    else:
        subject = f"Friendly note from {sender}"
        body = (
            f"Hi {receiver},\n\n"
            "It was nice hearing from you earlier. I hope your week is going well. "
            "I am sharing a quick hello and wishing you a calm evening.\n\n"
            f"Warm wishes,\n{sender}"
        )

    if data_type == "email":
        return _email_wrap(subject, body)
    if data_type == "job posting":
        return _job_wrap(subject, body)
    return body


def build_benign_multi_round(example: dict):
    role_bg = example.get("role_bg", {})
    sender = _first_name(role_bg.get("Sender"), "Jordan")
    receiver = _first_name(role_bg.get("Receiver"), "Alex")
    organization = _organization_name(role_bg.get("Sender_company"), "the team")
    category = str(example.get("category", "")).lower()

    if category == "fraudulent service":
        return [
            f"Hi {receiver}, this is {sender} from {organization}. I am sending a routine service update so you know your request is still in the normal queue.",
            "There is no action required from you today. Our staff will share a standard progress note after the review is completed.",
            "If you would like a copy of the public information sheet, I can send it in a later note.",
            "Thanks for your patience. I hope you have a smooth rest of the day.",
        ]
    if category == "impersonation":
        return [
            f"Hello {receiver}, this is {sender} with a regular notice from {organization}.",
            "We are arranging a routine planning session for next week and the calendar details will be shared separately.",
            "If there is a preferred meeting window on your side, feel free to mention morning or afternoon.",
            "Appreciate your time. I will close the loop once the schedule is finalized.",
        ]
    if category == "phishing":
        return [
            f"Hi {receiver}, this is a routine communication update from {organization}.",
            "Your newsletter and account preference settings remain unchanged, so no reply is needed.",
            "Our next message will simply contain general service news and travel tips for the coming month.",
            "Thank you for staying in touch with us. Wishing you a pleasant day ahead.",
        ]
    if category == "fake job posting":
        return [
            f"Hello {receiver}, thank you for your interest in our open position.",
            "The hiring team is reviewing applications this week and will contact shortlisted candidates with interview windows soon.",
            "For now there is nothing else you need to send unless we ask for a portfolio during the normal interview stage.",
            "Thanks again for your patience. We will be in touch after the first review round.",
        ]
    return [
        f"Hi {receiver}, it is nice to hear from you.",
        "I enjoy quiet weekends, cooking, and local walks, and I hope your week has been treating you well.",
        "If you would like to chat later about books or travel, I am happy to continue the conversation another time.",
        "Take care and have a relaxing evening.",
    ]
