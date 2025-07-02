INSERT INTO messages (session_id, role, content, metadata)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING id::text