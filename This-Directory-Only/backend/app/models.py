import uuid

from sqlalchemy import Column, Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    openai_api_key = Column(Text, nullable=True)
    mathpix_app_id = Column(Text, nullable=True)
    mathpix_app_key = Column(Text, nullable=True)

    jobs = relationship("Job", back_populates="user", cascade="all, delete-orphan")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    original_filename = Column(String(512), nullable=False)
    status = Column(String(32), nullable=False, default="queued")
    pdf_path = Column(Text, nullable=False)
    toc_csv_path = Column(Text, nullable=True)
    chapters_csv_path = Column(Text, nullable=True)
    chapters_dir = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="jobs")
    chapters = relationship("Chapter", back_populates="job", cascade="all, delete-orphan")
    artifacts = relationship("JobArtifact", back_populates="job", cascade="all, delete-orphan")


class Chapter(Base):
    __tablename__ = "chapters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    title = Column(String(512), nullable=False)
    start_page = Column(Integer, nullable=False)
    end_page = Column(Integer, nullable=False)
    filename = Column(String(512), nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    job = relationship("Job", back_populates="chapters")


class JobArtifact(Base):
    __tablename__ = "job_artifacts"
    __table_args__ = (
        UniqueConstraint("job_id", "artifact_key", name="job_artifacts_job_id_artifact_key_key"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False)
    artifact_key = Column(String(128), nullable=False)
    artifact_kind = Column(String(32), nullable=False, default="text")
    file_path = Column(Text, nullable=True)
    content_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    job = relationship("Job", back_populates="artifacts")


# ── Study groups ────────────────────────────────────────────────────────────
# A learner-facing group: a handful of people practising the same curriculum,
# reading each other's area mastery side by side. Modelled on Delta Note's
# `accountability_groups` (deployed-web/supabase/migrations/00023) — same three
# ways in (create / invite token / public directory), same one-group-per-person
# rule, same capability-token semantics.
#
# 🔴 THE JOIN TOKEN IS A CAPABILITY. Anyone holding it is in the group and can
# read every member's mastery. It is therefore minted with 128 bits of entropy,
# rotatable by the owner without disturbing the roster, and never returned by
# the public-directory read.

class StudyGroup(Base):
    __tablename__ = "study_groups"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(120), nullable=False)
    # 32 hex characters. Unique so a mint collision is a database error rather
    # than two groups sharing one door.
    join_token = Column(String(64), unique=True, index=True, nullable=False)
    owner_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    # 'private' (invite link only) or 'public' (listed in the directory).
    # Defaulted to the CLOSED state everywhere it is read, not to the last one
    # seen: a row whose listing state is unknown must be drawn as unlisted.
    visibility = Column(String(16), nullable=False, default="private")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    members = relationship(
        "StudyGroupMember", back_populates="group", cascade="all, delete-orphan"
    )


class StudyGroupMember(Base):
    __tablename__ = "study_group_members"
    __table_args__ = (
        # ONE GROUP PER PERSON. `mine` is singular on the client and on the
        # server, and a second membership would make "your group" ambiguous on
        # every surface that reads it. Enforced here rather than in the router
        # so a race between two join clicks cannot produce two rows.
        UniqueConstraint("user_id", name="study_group_members_user_id_key"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    group_id = Column(UUID(as_uuid=True), ForeignKey("study_groups.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    # 🔴 NEVER the email address. A roster is shown to everyone in the group,
    # and a group is joined by anyone holding a link; publishing addresses to it
    # would make an invite link a way of harvesting them.
    display_name = Column(String(120), nullable=False)
    joined_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    group = relationship("StudyGroup", back_populates="members")


# 🔴 A DAY CHECKLIST IS THE PERSON'S, NOT THE GROUP'S. It is keyed by
# (user_id, day) and never by group_id, so leaving a group and joining another
# carries your own accountability list with you instead of stranding it under a
# row that may have been deleted. The group only decides who may READ it: the
# day endpoints answer the checklists of the people in the caller's own group
# and of nobody else.
class StudyGroupDay(Base):
    __tablename__ = "study_group_days"
    __table_args__ = (
        # One row per person per day. The write path is an upsert-by-read, so a
        # second row would be a checklist that silently splits in two: one of
        # them gets written and the other one gets read.
        UniqueConstraint("user_id", "day", name="study_group_days_user_day_key"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    # A LOCAL calendar date, as the learner's own browser named it. Never
    # derived on the server from a timestamp: "today" for somebody eight hours
    # away is not the server's today, and a day written under the wrong key
    # reads as an empty list rather than as an error.
    day = Column(Date, nullable=False)
    # The Tiptap `{v, doc}` JSON string, byte-for-byte what Delta Note's
    # sub-goal editor stores. Text, not JSONB: this column is round-tripped, not
    # queried, and keeping it opaque is what lets the document schema change
    # without a migration.
    payload = Column(Text, nullable=False, default="")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
