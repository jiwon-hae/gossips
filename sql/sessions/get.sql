SELECT 
    id::text,
    user_id,
    metadata,
    created_at,
    updated_at,
    expires_at
FROM sessions
WHERE id = $1::uuid
AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)