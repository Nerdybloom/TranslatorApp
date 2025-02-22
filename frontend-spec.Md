# Frontend Technical spec

## Pages

### Dashboard '/dashboard'

The  purpose of this page is to provide a summary of learning and act as the default page when a user visits the web-app

This page contains the following components:
    -Last study Session
        shows last activity used
        when last activity used
        summarises wrong vs correct from last activity
        has a link to the group

    -Study progress
        -total words studied 
            -across all study session show the total words studied out of all possible words
            -display a mastery progress eg 60%
    -Quick stats
        -success rate eg 80%
        -total study session eg 4
        -total active groups eg 5
        -study streak eg 5 days
    -Start studing button
        -goes to study activities page

we ll need the following APIendpoints to power this page
GET /api /dashboard/last_study_session
GET /api /dashboard/study_progress
GET /api /dashboard/quick_stats


### Study Activity '/study-activities'

#### Purpose
the purpose of this page is to show a collection of study activities with a thumbnail and its name, to either launch or view the study activity.

#### Components

    - Study Activity Card
        -show a thumbnail of the study activity
        -the name of the study activity
        -a launch button to take us to the launch page
        -the view page to view more information about past study sessions for this study activity

#### Needed API Endpoints

-GET /api/study_activities

### study Activity s '/study_activities/:id

#### Purpose
The purpose of this page is to show the details of a study activity and its past study sessions

#### Components
    -Name of study activity
    -Thumbnail of study activities
    -Description of study activities
    -Launch button
    -study activities paginated list
        -id
        -activity name
        -group name
        -start time
        -end time(informed by the last word_review_item submitted)
        - number of review items

#### Needed API endpoints
-GET /api/study_activities/:id
-GET /api/study_activities/:id/study_sessions


### Study Activities '/study_activities/:id/launch'

#### Purpose
The purpose of this page is to launch a study activity

#### Components
-name of study activity
-launch form
    -select field for group
    -launch now button

## Behaviour
After the form is submitted a new tab opens with the study activity based on its URL provided in the database.

Also after form is submitted, the page will redirect to the study activity session show page

#### Needed API endpoints
- POST /api/study_activities

### Words '/words'

#### Purpose
The purpose of this is to show all words in our database

#### Components
-paginated word list
    -column
        -English
        -spanish
        -correct count 
        -wrong count
    -Pagination with 100 items per page
    -clicking the spanish word will take us to the word show page

#### Needed API Endpoint
-GET /api/words


### Word show '/words/:id'

#### Purpose
The purpose of this page is to show information about a specific word.

#### Components
-Spanish
-English
-Study Statistics
    -correct count
    -wrong count
-Word Groups
    -show as a series of pills eg tags
    -when group name is clicked, it will take us to the group show page

#### Needed API endpoint
-GET /api/words/:id

### Word Groups '/groups'

#### Purpose
The purpose of this page is to show a list of groups in our database

#### Components
-Paginated Group list
    -Columns
    -Group name
    -word count
-Clicking the group name will take us to the group show page

##### Needed API endpoints
GET /api/groups


### Group Show '/groups/:id'

#### Purpose
The purpose of this page is to show information about a specific group

#### Components
-Group Name
-Group Statistics
    -Total word count
-Words in Group (Paginated list of words)
    -should use the same components as the words index page
-study sessions (Paginated list of study sessions)
    -should use the same components as the study sessions index page

#### Needed API Endpoints
-GET /api/groups/:id (the name and groups stats)
-GET /api/groups/:id/words
-GET /api/groups/:id/study_sessions

### Study Sessions Index '/Study_sessions'

#### Purpose
The purpose of this page is to show a list of study sessions in our database

#### Components
-Paginated Study Session List
    -Columns
        -Id
        -Activity Name
        -Group Name
        -start Time
        -End time
        - Number of review items
    -clicking the study session id will take us to the study session show page

#### Needed API endpoints
GET /api/study_sessions

### Study Sessions show '/Study_sessions/:id'

#### Purpose
The purpose of this page is to show information about a specific study session

#### Components
-Study Session Details
    -Activity Name
    -Group Name
    -start Time
    -End time
    -Number of review items
-Words review items (Paginated list of words)
    -should be the same component as the words index page

    

#### Needed API endpoints
GET /api/study_sessions/:id
GET /api/study_sessions/:id/words

### Settings Page '/settings'

#### Purpose
The purpose of this page is to make configurations to the study portal

#### Components
-Theme selection eg light, Dark, System default
-Reset History Button
    -this will delete all study sessions and word review items
- Full reset button
    -this will drop all tables and re-create with seed data 

#### Needed API endpoints
POST /api/reset-history
POST /api/full_reset
