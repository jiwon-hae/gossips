INSERT INTO documents (title, celebrity, source, content, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id::text