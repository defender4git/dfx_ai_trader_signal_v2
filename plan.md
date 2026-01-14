# Mailing List Manager Development Plan

## Overview
Develop a comprehensive mailing list manager for trading signals with separate VIP and regular mailing lists. Provide CRUD operations for VIP emails and enable/disable functionality for regular users' email notifications.

## Todo List

1. **Create MailingList Database Model**
   - Add MailingList model to web_app.py with fields: id (primary key), email (unique), list_type ('vip'), is_active (boolean), created_at
   - Ensure proper database constraints

2. **Migrate VIP Emails from File to Database**
   - Add migration code in app startup to read static/vip_email.txt
   - Insert emails into MailingList table if they don't already exist
   - Handle validation and duplicate prevention

3. **Update Email Sending Logic**
   - Modify the signal alert sending in run_ai_analysis to query active VIP emails from MailingList instead of reading the file
   - Maintain existing functionality for regular users

4. **Create Mailing List Manager Route**
   - Add /mailing_list_manager route with @login_required decorator
   - Implement admin access check (current_user.is_admin)
   - Handle both GET (display) and POST (CRUD operations) requests

5. **Create Mailing List Manager Template**
   - Create templates/mailing_list_manager.html
   - Design UI with separate sections for VIP emails and regular users
   - Include tables displaying current emails/users with status
   - Add forms for adding new VIP emails
   - Include action buttons for edit, delete, and toggle operations

6. **Implement VIP Email CRUD Operations**
   - Add new VIP email: validate email format and uniqueness, insert into database
   - Edit existing VIP email: update email address with validation
   - Delete VIP email: remove from database
   - Toggle active status: enable/disable email for sending

7. **Implement Regular User Email Management**
   - Display all users with their current email_notifications status
   - Toggle email_notifications for individual users
   - Ensure only admins can modify user settings

8. **Add Validation and Error Handling**
   - Implement email format validation
   - Add flash messages for success/error feedback
   - Handle database errors gracefully
   - Ensure data integrity

## System Architecture
- **VIP Mailing List**: Stored in MailingList table with list_type='vip'
- **Regular Mailing List**: Uses existing User table with email_notifications flag
- **Access Control**: Admin-only access to mailing list manager
- **Email Sending**: Modified to use database queries for VIP emails
- **Migration**: One-time migration of existing VIP emails from file to database

## Dependencies
- Existing Flask, SQLAlchemy, and Flask-Login setup
- Admin user capability (is_admin flag in User model)
- Existing email sending infrastructure

## Success Criteria
- Admins can fully manage VIP email lists through web interface
- Admins can enable/disable email notifications for regular users
- Email alerts are sent to all active VIP emails and enabled regular users
- System maintains backwards compatibility
- Proper validation and error handling throughout