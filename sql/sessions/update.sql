UPDATE sessions
    SET metadata = metadata || $2::jsonb
    WHERE id = $1::uuid
    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)