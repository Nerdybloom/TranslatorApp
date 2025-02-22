## Business Goal: 
A language learning school wants to build a prototype of learning portal which will act as three things:
Inventory of possible vocabulary that can be learned
Act as a  Learning record store (LRS), providing correct and wrong score on practice vocabulary
A unified launchpad to launch different learning apps

You have been tasked with creating the backend API of the application.


## Technical specifications:
-Use SQLite3 as the database
-the backend will be built using python
-API will be built using Gin
-The API will always return JSON
-There will be no authentication or authorization
-Everything will be treated as a single user


## Database Schema:
we have the following tables:
-words: stored vocabulary words
    -id integer
    -english string
    -spanish string
    -parts json
    -created_at datetime

-words_groups: join table for words and groups many-to-many
    -id integer
    -word_id integer
    -group_id integer
    -created_at datetime

-groups: thematic groups of words
    -id integer
    -name string
    -created at datetime

-study_sessions: records of study sessions grouping word_review_items
    -id integer
    -group_id integer
    -created_at datetime

-study activities: a specific syudy activity linking a study session to group
    -id integer
    -study_session_id integer
    -group_id integer
    -created_at datetime

-word_review_items: a record of word practise, determinig if the word was correct or not.
    -word_id integer
    -study_session_id integer
    -correct boolean
    -created_at datetime

## API endpoints
    GET /api/dashboard/last_study_session
Response:
{
    "id": 1,
    "activity_name": "string",
    "group_name": "string",
    "start_time": "datetime",
    "end_time": "datetime",
    "correct_count": 10,
    "wrong_count": 5,
    "group_id": 1
}

GET /api/dashboard/study_progress
Response:
{
    "total_words": 100,
    "studied_words": 60,
    "mastery_percentage": 60
}

GET /api/dashboard/quick_stats
Response:
{
    "success_rate": 80,
    "total_study_sessions": 4,
    "active_groups": 5,
    "study_streak_days": 5
}

GET /api/dashboard/study_progress
GET /api/dashboard/quick_stats
GET /api/study_activities/:id
GET /api/study_activities/:id/study_sessions
POST /api/study_activities/
    -required params: group_id, study_activity_id
GET /api/words
    -pagination with 100 items per page
GET /api/words/:id
GET /api/groups
    -pagination with 100 items per page
GET /api/groups/:id
GET /api/groups/:id/words
GET /api/groups/:id/study_sessions
GET /api/study_sessions
GET /api/study_sessions/:id/words
POST /api/reset-history
POST /api/full_reset
POST /api/study_sessions/:id/words/:word_id/review
    -required params: correct