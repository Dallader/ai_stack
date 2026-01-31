# Baza Danych Studentów - Instrukcja

## Opis
System zawiera przykładową bazę danych studentów WSB Merito w Qdrant z ocenami, przedmiotami i punktami ECTS.

## Studenci w bazie danych

### 1. Jan Kowalski
- **Numer indeksu:** 12345
- **Email:** jan.kowalski@student.merito.pl
- **Rok studiów:** 2
- **Średnia ogólna:** 4.45
- **Semestry:** 1-4

### 2. Anna Nowak
- **Numer indeksu:** 12346
- **Email:** anna.nowak@student.merito.pl
- **Rok studiów:** 3
- **Średnia ogólna:** 4.81
- **Semestry:** 1-6 (4 lata studiów inżynierskich - ukończone)

### 3. Piotr Wiśniewski
- **Numer indeksu:** 12347
- **Email:** piotr.wisniewski@student.merito.pl
- **Rok studiów:** 1
- **Średnia ogólna:** 3.59
- **Semestry:** 1-2

### 4. Maria Lewandowska
- **Numer indeksu:** 12348
- **Email:** maria.lewandowska@student.merito.pl
- **Rok studiów:** 4
- **Średnia ogólna:** 4.28
- **Semestry:** 1-7

### 5. Katarzyna Dąbrowska
- **Numer indeksu:** 12349
- **Email:** katarzyna.dabrowska@student.merito.pl
- **Rok studiów:** 2
- **Średnia ogólna:** 4.65
- **Semestry:** 1-4

## Jak używać systemu

### 1. Logowanie do czatu

Gdy student wchodzi do czatu, system poprosi o:
1. **Imię** (np. Jan)
2. **Nazwisko** (np. Kowalski)
3. **Email** (np. jan.kowalski@student.merito.pl)
4. **Numer indeksu** (np. 12345)

**Ważne:** Dane muszą być zgodne z tymi w bazie danych, aby system mógł pobrać oceny studenta.

### 2. Pytania o oceny i średnie

Po zalogowaniu, student może zapytać o:

- **Średnią semestralną:**
  - "Jaka jest moja średnia z semestru 1?"
  - "Pokaż moją średnią za semestr 3"

- **Średnią roczną:**
  - "Jaka jest moja średnia za pierwszy rok?"
  - "Jaką mam średnią za rok akademicki 2?"

- **Średnią ogólną:**
  - "Jaka jest moja średnia za całe studia?"
  - "Pokaż moją średnią ogólną"
  - "Jaka jest moja średnia?"

- **Szczegółowe oceny:**
  - "Jakie oceny mam z semestru 2?"
  - "Pokaż moje wszystkie oceny"
  - "Ile punktów ECTS zdobyłem?"

### 3. Przykładowe sesje

**Przykład 1 - Student z 2 roku:**
```
System: Witaj! Jak masz na imię?
Student: Jan
System: Dziękuję! Jakie jest Twoje nazwisko?
Student: Kowalski
System: Świetnie! Podaj proszę swój adres email:
Student: jan.kowalski@student.merito.pl
System: Doskonale! Podaj swój numer indeksu:
Student: 12345
System: Dziękuję Jan! Twoje dane zostały zapisane. W czym mogę Ci pomóc?

Student: Jaka jest moja średnia ogólna?
System: [System pobierze dane i pokaże średnią 4.45 wraz ze szczegółami]

Student: Jaka jest moja średnia z semestru 3?
System: [System pokaże średnią 4.58 wraz z ocenami z poszczególnych przedmiotów]
```

**Przykład 2 - Obliczanie średniej za rok:**
```
Student: Jaka jest moja średnia za drugi rok studiów?
System: [System obliczy średnią ważoną z semestrów 3 i 4: 4.56]
```

## Endpointy API

### GET /student/{index_number}
Pobiera wszystkie dane studenta po numerze indeksu.

**Przykład:**
```bash
curl http://localhost:8002/student/12345
```

### POST /init-students
Inicjalizuje bazę danych studentów (usuwa starą i tworzy nową).

**Przykład:**
```bash
curl -X POST http://localhost:8002/init-students
```

## Obliczanie średnich

System automatycznie oblicza:

1. **Średnia semestralna** - średnia ważona wszystkich ocen w semestrze (wagi = punkty ECTS)
   ```
   średnia = Σ(ocena × ECTS) / Σ(ECTS)
   ```

2. **Średnia roczna** - średnia ważona z dwóch semestrów roku akademickiego
   - Rok 1 = semestry 1 i 2
   - Rok 2 = semestry 3 i 4
   - itd.

3. **Średnia ogólna** - średnia ważona ze wszystkich semestrów

## Struktura danych

Każdy student ma:
- Dane osobowe (imię, nazwisko, email, indeks)
- Program studiów
- Semestry z przedmiotami
- Dla każdego przedmiotu: nazwa, ocena (2.0-5.0), punkty ECTS
- Przeliczone średnie (semestralne, roczne, ogólne)

## Przykładowe przedmioty w systemie

- Matematyka (6 ECTS)
- Programowanie (6 ECTS)
- Algorytmy (6 ECTS)
- Bazy danych (6 ECTS)
- Sieci komputerowe (6 ECTS)
- Inżynieria oprogramowania (6 ECTS)
- Sztuczna inteligencja (5 ECTS)
- Bezpieczeństwo IT (6 ECTS)
- Machine Learning (6 ECTS)
- Cloud Computing (6 ECTS)
- Projekt dyplomowy (15 ECTS)
- i inne...

## Testowanie

Aby przetestować system:
1. Otwórz czat: http://localhost:8002
2. Zaloguj się danymi jednego z 5 studentów (podanymi powyżej)
3. Zapytaj o średnie, oceny, punkty ECTS
4. System automatycznie pobierze dane z Qdrant i obliczy wymagane średnie
