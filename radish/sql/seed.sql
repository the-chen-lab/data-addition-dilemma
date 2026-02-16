-- Create the locations table
CREATE TABLE IF NOT EXISTS locations (
  location_name TEXT PRIMARY KEY,
  location_order INTEGER NOT NULL
);

-- Create the items table
CREATE TABLE IF NOT EXISTS items (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name TEXT NOT NULL,
  location TEXT NOT NULL REFERENCES locations(location_name),
  order_source TEXT NOT NULL DEFAULT 'Order from anywhere',
  short_list BOOLEAN NOT NULL DEFAULT false
);

-- Enable Row Level Security
ALTER TABLE locations ENABLE ROW LEVEL SECURITY;
ALTER TABLE items ENABLE ROW LEVEL SECURITY;

-- Allow public read access
CREATE POLICY "Allow public read access"
  ON locations FOR SELECT TO anon USING (true);

CREATE POLICY "Allow public read access"
  ON items FOR SELECT TO anon USING (true);

-- Seed locations
INSERT INTO locations (location_name, location_order) VALUES
  ('Counter', 1),
  ('Produce', 2),
  ('Cabinet', 3),
  ('Fridge 1', 4),
  ('Fridge 2', 5),
  ('Freezer', 6);

-- Seed items
INSERT INTO items (name, location, order_source, short_list) VALUES
  -- Counter
  ('Bananas', 'Counter', 'Order from anywhere', true),
  ('Avocados', 'Counter', 'Order from anywhere', true),
  ('Bread', 'Counter', 'Order from anywhere', true),
  ('Olive Oil', 'Counter', 'Order from anywhere', false),

  -- Produce
  ('Spinach', 'Produce', 'Order from anywhere', true),
  ('Tomatoes', 'Produce', 'Order from anywhere', true),
  ('Lemons', 'Produce', 'Order from anywhere', false),
  ('Garlic', 'Produce', 'Order from anywhere', false),
  ('Onions', 'Produce', 'Order from anywhere', false),

  -- Cabinet
  ('Rice', 'Cabinet', 'Order from anywhere', false),
  ('Pasta', 'Cabinet', 'Order from anywhere', false),
  ('Canned Tomatoes', 'Cabinet', 'Order from anywhere', false),
  ('Soy Sauce', 'Cabinet', 'Order from anywhere', false),
  ('Peanut Butter', 'Cabinet', 'Order from anywhere', false),
  ('Honey', 'Cabinet', 'Order from anywhere', false),

  -- Fridge 1
  ('Milk', 'Fridge 1', 'Order from anywhere', true),
  ('Eggs', 'Fridge 1', 'Order from anywhere', true),
  ('Butter', 'Fridge 1', 'Order from anywhere', true),
  ('Yogurt', 'Fridge 1', 'Order from anywhere', false),
  ('Cheese', 'Fridge 1', 'Order from anywhere', false),

  -- Fridge 2
  ('Tofu', 'Fridge 2', 'Order from anywhere', false),
  ('Kimchi', 'Fridge 2', 'H Mart', false),
  ('Miso Paste', 'Fridge 2', 'H Mart', false),
  ('Hot Sauce', 'Fridge 2', 'Order from anywhere', false),

  -- Freezer
  ('Frozen Dumplings', 'Freezer', 'H Mart', false),
  ('Ice Cream', 'Freezer', 'Order from anywhere', false),
  ('Frozen Peas', 'Freezer', 'Order from anywhere', false),
  ('Frozen Berries', 'Freezer', 'Trader Joe''s', false);
