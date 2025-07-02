INSERT INTO chunks (document_id, content, embedding, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3::vector, $4, $5, $6)