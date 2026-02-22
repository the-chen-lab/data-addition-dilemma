-- Run this once in the Supabase SQL editor to allow the edit page to write to the items table.
-- (locations are edited manually via SQL, so no write policy needed there.)

CREATE POLICY "Allow anon insert"
  ON items FOR INSERT TO anon WITH CHECK (true);

CREATE POLICY "Allow anon update"
  ON items FOR UPDATE TO anon USING (true) WITH CHECK (true);

CREATE POLICY "Allow anon delete"
  ON items FOR DELETE TO anon USING (true);
