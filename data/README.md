# Google Analitics
## Dataset
Each event has many field that contain information that describe what happen with the app in a particular moment.
- user_pseudo_id: unique identifier of a user 
- event_timestamp: the moment in which the event was triger
- device: information about the divice (OS, model, language)
- geo: geo information (country)
- ecommerce
- event_name: type of event
- event_params: information related to event_name

A user could have different demographic information, for simplicity and also for la problem to solve we goint to take the information from the engagement event

## Events

| Event                                                   | Trigger...                                                         | Parameters                              |
|---------------------------------------------------------|--------------------------------------------------------------------|-----------------------------------------|
| <span style="color:red">ad_reward</span>                |                                                                    |                                         |
| app_clear_data                                          |                                                                    |                                         |
| app_exception                                           |                                                                    |                                         |
| app_remove                                              |                                                                    |                                         |
| app_update                                              |                                                                    |                                         |
| <span style="color:red">challenge_a_friend</span>       |                                                                    |                                         |
| challenge_accepted                                      |                                                                    |                                         |
| <span style="color:red">completed_5_levels</span>       |                                                                    |                                         |
| dynamic_link_app_open                                   |                                                                    |                                         |
| dynamic_link_first_open                                 |                                                                    |                                         |
| error                                                   |                                                                    |                                         |
| firebase_campaign                                       |                                                                    |                                         |
| first_open                                              |                                                                    |                                         |
| in_app_purchase                                         |                                                                    |                                         |
| level_complete                                          |                                                                    |                                         |
| <span style="color:red">level_complete_quickplay</span> |                                                                    |                                         |
| level_end                                               | when a user completes a level in the game                          | level_name, success                     |
| <span style="color:red">level_end_quickplay</span>      |                                                                    |                                         |
| level_fail                                              |                                                                    |                                         |
| level_fail_quickplay                                    |                                                                    |                                         |
| level_reset                                             |                                                                    |                                         |
| <span style="color:red">level_reset_quickplay</span>    |                                                                    |                                         |
| level_retry                                             |                                                                    |                                         |
| level_retry_quickplay                                   |                                                                    |                                         |
| level_start                                             | when a user starts a new level in the game                         | level_name                              |
| <span style="color:red">level_start_quickplay</span>    |                                                                    |                                         |
| level_up                                                | when a player levels-up in the game                                | character, level                        |
| no_more_extra_steps                                     |                                                                    |                                         |
| notification_foreground                                 |                                                                    |                                         |
| os_update                                               |                                                                    |                                         |
| <span style="color:red">post_score</span>               | when a player posts his or her score                               | level, character, score                 |
| screen_view                                             |                                                                    |                                         |
| select_content                                          | when a user has selected content                                   | content_type, item_id                   |
| session_start                                           |                                                                    |                                         |
| <span style="color:red">spend_virtual_currency</span>   | when a user has spent virtual currency (coins, gems, tokens, etc.) | item_name, virtual_currency_name, value |
| <span style="color:red">use_extra_steps</span>          |                                                                    |                                         |
| <span style="color:red">user_engagement</span>          |                                                                    |                                         |

