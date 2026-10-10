from datetime import datetime, timezone

from simon.pool import get_pool
from simon.records import AgentProfile, Lead, PropertyCard, ViewingSlot

PROPERTY_FIELDS = ",".join(
    [
        "id",
        "title",
        "description",
        "property_type",
        "listing_type",
        "price_period",
        "status",
        "price",
        "currency",
        "bedrooms",
        "bathrooms",
        "square_meters",
        "lot_size_sqm",
        "location",
        "address",
        "town",
        "city",
        "county",
        "country",
        "amenities",
        "furnished",
        "year_built",
        "video_url",
        "agent_id",
        "external_id",
    ]
)


def as_card(row: dict) -> PropertyCard:
    return PropertyCard(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        property_type=row["property_type"],
        listing_type=row["listing_type"],
        price_period=row["price_period"],
        status=row["status"],
        price=float(row["price"]),
        currency=row["currency"],
        bedrooms=row.get("bedrooms"),
        bathrooms=row.get("bathrooms"),
        square_meters=(
            float(row["square_meters"]) if row.get("square_meters") else None
        ),
        lot_size_sqm=(
            float(row["lot_size_sqm"]) if row.get("lot_size_sqm") else None
        ),
        location=row["location"],
        address=row.get("address"),
        town=row.get("town"),
        city=row.get("city", "Nairobi"),
        county=row.get("county"),
        country=row.get("country", "Kenya"),
        amenities=row.get("amenities") or [],
        furnished=row.get("furnished") or False,
        year_built=row.get("year_built"),
        video_url=row.get("video_url"),
        agent_id=row.get("agent_id"),
        external_id=row.get("external_id"),
    )


async def browse(
    *,
    deal: str | None = None,
    kind: str | None = None,
    status: str = "available",
    min_price: float | None = None,
    max_price: float | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    city: str | None = None,
    area: str | None = None,
    amenity: str | None = None,
    limit: int = 6,
    page: int = 0,
) -> list[PropertyCard]:
    pool = await get_pool()
    q = pool.from_("properties").select(PROPERTY_FIELDS).eq("status", status)
    if deal:
        q = q.eq("listing_type", deal)
    if kind:
        q = q.eq("property_type", kind)
    if min_price is not None:
        q = q.gte("price", min_price)
    if max_price is not None:
        q = q.lte("price", max_price)
    if bedrooms is not None:
        q = q.gte("bedrooms", bedrooms)
    if bathrooms is not None:
        q = q.gte("bathrooms", bathrooms)
    if city:
        q = q.ilike("city", f"%{city}%")
    if area:
        q = q.ilike("location", f"%{area}%")
    if amenity:
        q = q.cs("amenities", [amenity])
    rows = (
        await q.order("created_at", desc=True)
        .range(page * limit, page * limit + limit - 1)
        .execute()
    ).data
    return [as_card(r) for r in rows]


async def fetch_card(property_id: str) -> PropertyCard | None:
    pool = await get_pool()
    rows = (
        await pool.from_("properties")
        .select(PROPERTY_FIELDS)
        .eq("id", property_id)
        .limit(1)
        .execute()
    ).data
    return as_card(rows[0]) if rows else None


async def fetch_cards(property_ids: list[str]) -> list[PropertyCard]:
    if not property_ids:
        return []
    pool = await get_pool()
    rows = (
        await pool.from_("properties")
        .select(PROPERTY_FIELDS)
        .in_("id", property_ids)
        .execute()
    ).data
    by_id = {r["id"]: as_card(r) for r in rows}
    return [by_id[i] for i in property_ids if i in by_id]


async def gallery_for(property_id: str) -> list[str]:
    pool = await get_pool()
    rows = (
        await pool.from_("property_images")
        .select("url")
        .eq("property_id", property_id)
        .order("sort_order")
        .execute()
    ).data
    return [r["url"] for r in rows]


async def agent_for(agent_id: str) -> AgentProfile | None:
    pool = await get_pool()
    rows = (
        await pool.from_("agents")
        .select(
            "id,first_name,last_name,email,phone,agency_name,is_verified"
        )
        .eq("id", agent_id)
        .limit(1)
        .execute()
    ).data
    if not rows:
        return None
    r = rows[0]
    return AgentProfile(
        id=r["id"],
        first_name=r["first_name"],
        last_name=r.get("last_name"),
        email=r.get("email"),
        phone=r.get("phone"),
        agency_name=r.get("agency_name"),
        is_verified=r.get("is_verified") or False,
    )


async def post_listing(
    *,
    title: str,
    description: str,
    property_type: str,
    listing_type: str,
    price: float,
    location: str,
    city: str = "Nairobi",
    county: str | None = None,
    bedrooms: int | None = None,
    bathrooms: int | None = None,
    square_meters: float | None = None,
    lot_size_sqm: float | None = None,
    address: str | None = None,
    town: str | None = None,
    amenities: list[str] | None = None,
    furnished: bool = False,
    price_period: str = "one_time",
    year_built: int | None = None,
    agent_id: str | None = None,
) -> PropertyCard:
    payload = {
        "title": title,
        "description": description,
        "property_type": property_type,
        "listing_type": listing_type,
        "price": price,
        "price_period": price_period,
        "location": location,
        "city": city,
        "country": "Kenya",
        "furnished": furnished,
        "amenities": amenities or [],
    }
    for key, value in {
        "county": county,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "square_meters": square_meters,
        "lot_size_sqm": lot_size_sqm,
        "address": address,
        "town": town,
        "year_built": year_built,
        "agent_id": agent_id,
    }.items():
        if value is not None:
            payload[key] = value
    pool = await get_pool()
    rows = (
        await pool.from_("properties").insert(payload).execute()
    ).data
    return as_card(rows[0])


async def book_viewing(
    *,
    property_id: str,
    customer_phone: str,
    viewing_date: str,
    customer_name: str | None = None,
    duration_minutes: int = 30,
    notes: str | None = None,
) -> ViewingSlot:
    pool = await get_pool()
    rows = (
        await pool.from_("scheduled_viewings")
        .insert(
            {
                "property_id": property_id,
                "customer_phone": customer_phone,
                "customer_name": customer_name,
                "viewing_date": viewing_date,
                "duration_minutes": duration_minutes,
                "notes": notes,
                "status": "confirmed",
            }
        )
        .execute()
    ).data
    r = rows[0]
    return ViewingSlot(
        id=r["id"],
        property_id=r["property_id"],
        customer_phone=r["customer_phone"],
        customer_name=r.get("customer_name"),
        viewing_date=r["viewing_date"],
        duration_minutes=r["duration_minutes"],
        status=r["status"],
        notes=r.get("notes"),
    )


async def viewings_for(phone: str) -> list[ViewingSlot]:
    pool = await get_pool()
    rows = (
        await pool.from_("scheduled_viewings")
        .select(
            "id,property_id,customer_phone,customer_name,viewing_date,"
            "duration_minutes,status,notes"
        )
        .eq("customer_phone", phone)
        .order("viewing_date", desc=False)
        .execute()
    ).data
    return [
        ViewingSlot(
            id=r["id"],
            property_id=r.get("property_id"),
            customer_phone=r["customer_phone"],
            customer_name=r.get("customer_name"),
            viewing_date=r["viewing_date"],
            duration_minutes=r["duration_minutes"],
            status=r["status"],
            notes=r.get("notes"),
        )
        for r in rows
    ]


async def save_lead(
    *,
    customer_phone: str,
    inquiry_type: str = "general",
    customer_name: str | None = None,
    customer_email: str | None = None,
    property_id: str | None = None,
    agent_id: str | None = None,
    message: str | None = None,
) -> Lead:
    pool = await get_pool()
    rows = (
        await pool.from_("property_inquiries")
        .insert(
            {
                "customer_phone": customer_phone,
                "customer_name": customer_name,
                "customer_email": customer_email,
                "property_id": property_id,
                "agent_id": agent_id,
                "inquiry_type": inquiry_type,
                "message": message,
            }
        )
        .execute()
    ).data
    r = rows[0]
    return Lead(
        id=r["id"],
        customer_phone=r["customer_phone"],
        customer_name=r.get("customer_name"),
        customer_email=r.get("customer_email"),
        property_id=r.get("property_id"),
        agent_id=r.get("agent_id"),
        inquiry_type=r["inquiry_type"],
        message=r.get("message"),
    )


def normalize_when(day: str, hour: str) -> str:
    when = datetime.fromisoformat(f"{day}T{hour}")
    return when.astimezone(timezone.utc).isoformat()
