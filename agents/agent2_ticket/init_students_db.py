"""
Initialize student database in Qdrant with sample data
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer
import json

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "students"

# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL)
embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Sample students data
STUDENTS_DATA = [
    {
        "student_id": "12345",
        "index_number": "12345",
        "name": "Jan",
        "surname": "Kowalski",
        "email": "jan.kowalski@student.merito.pl",
        "program": "Informatyka",
        "year": 2,
        "semesters": [
            {
                "semester": 1,
                "year": 2023,
                "subjects": [
                    {"name": "Matematyka", "grade": 4.0, "ects": 6},
                    {"name": "Programowanie", "grade": 5.0, "ects": 6},
                    {"name": "Fizyka", "grade": 3.5, "ects": 4},
                    {"name": "Języki angielski", "grade": 4.5, "ects": 2},
                ]
            },
            {
                "semester": 2,
                "year": 2024,
                "subjects": [
                    {"name": "Algorytmy", "grade": 4.5, "ects": 6},
                    {"name": "Bazy danych", "grade": 5.0, "ects": 6},
                    {"name": "Systemy operacyjne", "grade": 4.0, "ects": 5},
                    {"name": "Statystyka", "grade": 3.5, "ects": 3},
                ]
            },
            {
                "semester": 3,
                "year": 2024,
                "subjects": [
                    {"name": "Sieci komputerowe", "grade": 4.5, "ects": 6},
                    {"name": "Inżynieria oprogramowania", "grade": 5.0, "ects": 6},
                    {"name": "Sztuczna inteligencja", "grade": 4.5, "ects": 5},
                    {"name": "Grafika komputerowa", "grade": 4.0, "ects": 3},
                ]
            },
            {
                "semester": 4,
                "year": 2025,
                "subjects": [
                    {"name": "Bezpieczeństwo IT", "grade": 4.5, "ects": 6},
                    {"name": "Technologie webowe", "grade": 5.0, "ects": 6},
                    {"name": "Zarządzanie projektami", "grade": 4.0, "ects": 4},
                    {"name": "Marketing", "grade": 4.5, "ects": 4},
                ]
            }
        ]
    },
    {
        "student_id": "12346",
        "index_number": "12346",
        "name": "Anna",
        "surname": "Nowak",
        "email": "anna.nowak@student.merito.pl",
        "program": "Informatyka",
        "year": 3,
        "semesters": [
            {
                "semester": 1,
                "year": 2022,
                "subjects": [
                    {"name": "Matematyka", "grade": 5.0, "ects": 6},
                    {"name": "Programowanie", "grade": 5.0, "ects": 6},
                    {"name": "Fizyka", "grade": 4.5, "ects": 4},
                    {"name": "Języki angielski", "grade": 5.0, "ects": 2},
                ]
            },
            {
                "semester": 2,
                "year": 2023,
                "subjects": [
                    {"name": "Algorytmy", "grade": 5.0, "ects": 6},
                    {"name": "Bazy danych", "grade": 4.5, "ects": 6},
                    {"name": "Systemy operacyjne", "grade": 5.0, "ects": 5},
                    {"name": "Statystyka", "grade": 4.0, "ects": 3},
                ]
            },
            {
                "semester": 3,
                "year": 2023,
                "subjects": [
                    {"name": "Sieci komputerowe", "grade": 4.5, "ects": 6},
                    {"name": "Inżynieria oprogramowania", "grade": 5.0, "ects": 6},
                    {"name": "Sztuczna inteligencja", "grade": 5.0, "ects": 5},
                    {"name": "Grafika komputerowa", "grade": 4.5, "ects": 3},
                ]
            },
            {
                "semester": 4,
                "year": 2024,
                "subjects": [
                    {"name": "Bezpieczeństwo IT", "grade": 5.0, "ects": 6},
                    {"name": "Technologie webowe", "grade": 5.0, "ects": 6},
                    {"name": "Zarządzanie projektami", "grade": 4.5, "ects": 4},
                    {"name": "Marketing", "grade": 4.0, "ects": 4},
                ]
            },
            {
                "semester": 5,
                "year": 2024,
                "subjects": [
                    {"name": "Machine Learning", "grade": 5.0, "ects": 6},
                    {"name": "Cloud Computing", "grade": 5.0, "ects": 6},
                    {"name": "Seminarium dyplomowe", "grade": 5.0, "ects": 4},
                    {"name": "Blockchain", "grade": 4.5, "ects": 4},
                ]
            },
            {
                "semester": 6,
                "year": 2025,
                "subjects": [
                    {"name": "Projekt dyplomowy", "grade": 5.0, "ects": 15},
                    {"name": "Etyka zawodowa", "grade": 5.0, "ects": 2},
                    {"name": "Startup", "grade": 4.5, "ects": 3},
                ]
            }
        ]
    },
    {
        "student_id": "12347",
        "index_number": "12347",
        "name": "Piotr",
        "surname": "Wiśniewski",
        "email": "piotr.wisniewski@student.merito.pl",
        "program": "Informatyka",
        "year": 1,
        "semesters": [
            {
                "semester": 1,
                "year": 2024,
                "subjects": [
                    {"name": "Matematyka", "grade": 3.5, "ects": 6},
                    {"name": "Programowanie", "grade": 4.0, "ects": 6},
                    {"name": "Fizyka", "grade": 3.0, "ects": 4},
                    {"name": "Języki angielski", "grade": 4.0, "ects": 2},
                ]
            },
            {
                "semester": 2,
                "year": 2025,
                "subjects": [
                    {"name": "Algorytmy", "grade": 3.5, "ects": 6},
                    {"name": "Bazy danych", "grade": 4.0, "ects": 6},
                    {"name": "Systemy operacyjne", "grade": 3.5, "ects": 5},
                    {"name": "Statystyka", "grade": 3.0, "ects": 3},
                ]
            }
        ]
    },
    {
        "student_id": "12348",
        "index_number": "12348",
        "name": "Maria",
        "surname": "Lewandowska",
        "email": "maria.lewandowska@student.merito.pl",
        "program": "Informatyka",
        "year": 4,
        "semesters": [
            {
                "semester": 1,
                "year": 2021,
                "subjects": [
                    {"name": "Matematyka", "grade": 4.5, "ects": 6},
                    {"name": "Programowanie", "grade": 4.0, "ects": 6},
                    {"name": "Fizyka", "grade": 4.0, "ects": 4},
                    {"name": "Języki angielski", "grade": 5.0, "ects": 2},
                ]
            },
            {
                "semester": 2,
                "year": 2022,
                "subjects": [
                    {"name": "Algorytmy", "grade": 4.0, "ects": 6},
                    {"name": "Bazy danych", "grade": 4.5, "ects": 6},
                    {"name": "Systemy operacyjne", "grade": 4.0, "ects": 5},
                    {"name": "Statystyka", "grade": 4.0, "ects": 3},
                ]
            },
            {
                "semester": 3,
                "year": 2022,
                "subjects": [
                    {"name": "Sieci komputerowe", "grade": 4.0, "ects": 6},
                    {"name": "Inżynieria oprogramowania", "grade": 4.5, "ects": 6},
                    {"name": "Sztuczna inteligencja", "grade": 4.0, "ects": 5},
                    {"name": "Grafika komputerowa", "grade": 3.5, "ects": 3},
                ]
            },
            {
                "semester": 4,
                "year": 2023,
                "subjects": [
                    {"name": "Bezpieczeństwo IT", "grade": 4.5, "ects": 6},
                    {"name": "Technologie webowe", "grade": 4.0, "ects": 6},
                    {"name": "Zarządzanie projektami", "grade": 4.0, "ects": 4},
                    {"name": "Marketing", "grade": 4.5, "ects": 4},
                ]
            },
            {
                "semester": 5,
                "year": 2023,
                "subjects": [
                    {"name": "Machine Learning", "grade": 4.5, "ects": 6},
                    {"name": "Cloud Computing", "grade": 4.0, "ects": 6},
                    {"name": "Seminarium dyplomowe", "grade": 4.5, "ects": 4},
                    {"name": "Blockchain", "grade": 4.0, "ects": 4},
                ]
            },
            {
                "semester": 6,
                "year": 2024,
                "subjects": [
                    {"name": "Projekt dyplomowy", "grade": 4.5, "ects": 15},
                    {"name": "Etyka zawodowa", "grade": 5.0, "ects": 2},
                    {"name": "Startup", "grade": 4.0, "ects": 3},
                ]
            },
            {
                "semester": 7,
                "year": 2024,
                "subjects": [
                    {"name": "Zaawansowane algorytmy", "grade": 4.0, "ects": 6},
                    {"name": "IoT", "grade": 4.5, "ects": 6},
                    {"name": "Praktyki zawodowe", "grade": 5.0, "ects": 6},
                    {"name": "Prawo IT", "grade": 4.0, "ects": 2},
                ]
            }
        ]
    },
    {
        "student_id": "12349",
        "index_number": "12349",
        "name": "Katarzyna",
        "surname": "Dąbrowska",
        "email": "katarzyna.dabrowska@student.merito.pl",
        "program": "Informatyka",
        "year": 2,
        "semesters": [
            {
                "semester": 1,
                "year": 2023,
                "subjects": [
                    {"name": "Matematyka", "grade": 5.0, "ects": 6},
                    {"name": "Programowanie", "grade": 4.5, "ects": 6},
                    {"name": "Fizyka", "grade": 4.0, "ects": 4},
                    {"name": "Języki angielski", "grade": 4.5, "ects": 2},
                ]
            },
            {
                "semester": 2,
                "year": 2024,
                "subjects": [
                    {"name": "Algorytmy", "grade": 5.0, "ects": 6},
                    {"name": "Bazy danych", "grade": 4.5, "ects": 6},
                    {"name": "Systemy operacyjne", "grade": 4.5, "ects": 5},
                    {"name": "Statystyka", "grade": 4.0, "ects": 3},
                ]
            },
            {
                "semester": 3,
                "year": 2024,
                "subjects": [
                    {"name": "Sieci komputerowe", "grade": 5.0, "ects": 6},
                    {"name": "Inżynieria oprogramowania", "grade": 4.5, "ects": 6},
                    {"name": "Sztuczna inteligencja", "grade": 5.0, "ects": 5},
                    {"name": "Grafika komputerowa", "grade": 4.5, "ects": 3},
                ]
            },
            {
                "semester": 4,
                "year": 2025,
                "subjects": [
                    {"name": "Bezpieczeństwo IT", "grade": 5.0, "ects": 6},
                    {"name": "Technologie webowe", "grade": 4.5, "ects": 6},
                    {"name": "Zarządzanie projektami", "grade": 4.5, "ects": 4},
                    {"name": "Marketing", "grade": 4.0, "ects": 4},
                ]
            }
        ]
    }
]

def calculate_semester_average(subjects):
    """Calculate weighted average for a semester"""
    total_points = sum(s["grade"] * s["ects"] for s in subjects)
    total_ects = sum(s["ects"] for s in subjects)
    return round(total_points / total_ects, 2) if total_ects > 0 else 0

def calculate_year_average(semesters_data, year_num):
    """Calculate average for academic year (2 semesters)"""
    semester_1 = year_num * 2 - 1
    semester_2 = year_num * 2
    
    relevant_semesters = [s for s in semesters_data if s["semester"] in [semester_1, semester_2]]
    
    if not relevant_semesters:
        return 0
    
    all_subjects = []
    for sem in relevant_semesters:
        all_subjects.extend(sem["subjects"])
    
    return calculate_semester_average(all_subjects)

def calculate_total_average(semesters_data):
    """Calculate average for all semesters"""
    all_subjects = []
    for sem in semesters_data:
        all_subjects.extend(sem["subjects"])
    
    return calculate_semester_average(all_subjects)

def create_student_text_representation(student):
    """Create text representation for embedding"""
    text_parts = [
        f"Student: {student['name']} {student['surname']}",
        f"Email: {student['email']}",
        f"Numer indeksu: {student['index_number']}",
        f"Program: {student['program']}",
        f"Rok studiów: {student['year']}"
    ]
    
    for semester in student["semesters"]:
        avg = calculate_semester_average(semester["subjects"])
        text_parts.append(f"Semestr {semester['semester']} ({semester['year']}): średnia {avg}")
        for subject in semester["subjects"]:
            text_parts.append(f"  - {subject['name']}: {subject['grade']} ({subject['ects']} ECTS)")
    
    return "\n".join(text_parts)

def init_students_collection():
    """Initialize students collection in Qdrant"""
    try:
        # Try to delete existing collection
        try:
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            pass
        
        # Create new collection
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest_models.VectorParams(
                size=1024,  # BAAI/bge-large-en-v1.5 dimension
                distance=rest_models.Distance.COSINE
            )
        )
        print(f"Created collection: {COLLECTION_NAME}")
        
        # Add students
        points = []
        for student in STUDENTS_DATA:
            # Calculate averages
            student["semester_averages"] = [
                {
                    "semester": sem["semester"],
                    "average": calculate_semester_average(sem["subjects"])
                }
                for sem in student["semesters"]
            ]
            
            # Calculate year averages
            max_year = (len(student["semesters"]) + 1) // 2
            student["year_averages"] = [
                {
                    "year": y,
                    "average": calculate_year_average(student["semesters"], y)
                }
                for y in range(1, max_year + 1)
            ]
            
            # Calculate total average
            student["total_average"] = calculate_total_average(student["semesters"])
            
            # Create text representation and embedding
            text = create_student_text_representation(student)
            vector = embedding_model.encode(text).tolist()
            
            # Create point
            point = rest_models.PointStruct(
                id=int(student["student_id"]),  # Convert to integer for Qdrant
                vector=vector,
                payload=student
            )
            points.append(point)
        
        # Upload to Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        print(f"Successfully initialized {len(points)} students in Qdrant")
        print(f"\nSample students:")
        for student in STUDENTS_DATA[:3]:
            print(f"  - {student['name']} {student['surname']} (Index: {student['index_number']})")
            print(f"    Email: {student['email']}")
            print(f"    Total average: {student['total_average']}")
        
    except Exception as e:
        print(f"Error initializing students database: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Initializing students database...")
    init_students_collection()
