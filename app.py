import os
import json
from datetime import date
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

DATA_FILE = "students.json"
MODEL_ID = os.getenv("HF_MODEL_ID")
TOKEN = os.getenv("HF_TOKEN")


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def attendance_percentage(student):
    records = list(student["attendance"].values())
    if not records:
        return 0.0
    present = sum(value == "Present" for value in records)
    return round(present / len(records) * 100, 2)


def show_report(data):
    percentages = []
    print("\n--- Attendance Report ---")
    for student in data["students"]:
        percentage = attendance_percentage(student)
        percentages.append(percentage)
        print(
            f'{student["name"]} (Roll {student["roll_no"]}): '
            f'{percentage}%'
        )

    average = round(sum(percentages) / len(percentages), 2) if percentages else 0
    print(f"\nClass average attendance: {average}%")
    return average


def ask_ai(question, data):
    if not TOKEN or not MODEL_ID:
        print("HF_TOKEN or HF_MODEL_ID missing in .env")
        return

    client = InferenceClient(
        model=MODEL_ID,
        token=TOKEN
    )

    prompt = f"""
You are an attendance management assistant.
Use only the supplied student data. Never invent attendance records.
Reply with valid JSON only, in this format:
{{"answer": "short answer", "changes": []}}

If the user requests an attendance change, use changes like:
{{"roll_no": "201", "date": "2026-10-08", "status": "Present"}}
For removing a record, use:
{{"roll_no": "201", "date": "2026-10-08", "status": "REMOVE"}}
For questions that do not change data, changes must be [].
Allowed dates: 2026-10-08 through 2026-10-14.
Allowed statuses: Present, Absent, REMOVE.

Student data:
{json.dumps(data, ensure_ascii=False)}

User request: {question}
"""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400
        )
        text = response.choices[0].message.content
        result = json.loads(text)

        # Validate changes before saving anything.
        valid_dates = {
            "2026-10-08", "2026-10-09", "2026-10-10",
            "2026-10-11", "2026-10-12", "2026-10-13",
            "2026-10-14"
        }

        for change in result.get("changes", []):
            roll = str(change.get("roll_no", ""))
            day = change.get("date", "")
            status = change.get("status", "")

            student = next(
                (s for s in data["students"] if s["roll_no"] == roll),
                None
            )

            if not student or day not in valid_dates:
                raise ValueError("AI returned an unknown student or date.")
            if status not in ("Present", "Absent", "REMOVE"):
                raise ValueError("AI returned an invalid attendance status.")

        for change in result.get("changes", []):
            student = next(
                s for s in data["students"]
                if s["roll_no"] == str(change["roll_no"])
            )
            day = change["date"]
            if change["status"] == "REMOVE":
                student["attendance"].pop(day, None)
            else:
                student["attendance"][day] = change["status"]

        if result.get("changes"):
            save_data(data)
            print("students.json updated successfully.")

        print("\nAI response:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        show_report(data)

    except Exception as error:
        print(f"AI request failed: {error}")
        print("Your JSON file was not changed by this failed request.")


def main():
    data = load_data()

    while True:
        print("\n=== AI Attendance Management ===")
        print("1. Ask AI / manage attendance")
        print("2. Show attendance report")
        print("3. Exit")

        choice = input("Choose (1-3): ").strip()

        if choice == "1":
            question = input("Enter your question or attendance change: ")
            ask_ai(question, data)
            data = load_data()
        elif choice == "2":
            show_report(data)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()