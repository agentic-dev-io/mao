"""
Tests for the storage module (DuckDB-based vector stores).
"""

import pytest


@pytest.mark.asyncio
async def test_knowledge_tree_basic_operations(knowledge_tree):
    point_id = await knowledge_tree.add_entry_async(
        text="This is a test knowledge point",
        tags=["test", "importance:high"],
    )

    assert point_id is not None
    assert isinstance(point_id, str)

    results = await knowledge_tree.search_async("test knowledge")
    assert len(results) > 0
    assert any(r["id"] == point_id for r in results)

    await knowledge_tree.delete_entry_async(point_id)

    results = await knowledge_tree.search_async("test knowledge")
    assert not any(r["id"] == point_id for r in results)


@pytest.mark.asyncio
async def test_experience_tree_basic_operations(experience_tree):
    point_id = await experience_tree.add_entry_async(
        text="This is a test experience",
        tags=["outcome:success", "importance:high"],
    )

    assert point_id is not None
    assert isinstance(point_id, str)

    results = await experience_tree.search_async("test experience")
    assert len(results) > 0
    assert any(r["id"] == point_id for r in results)

    await experience_tree.delete_entry_async(point_id)

    results = await experience_tree.search_async("test experience")
    assert not any(r["id"] == point_id for r in results)


@pytest.mark.asyncio
async def test_knowledge_tree_batch_operations(knowledge_tree):
    texts = [f"Knowledge point {i}" for i in range(5)]
    tags_list = [[f"index:{i}"] for i in range(5)]

    point_ids = await knowledge_tree.add_entries_batch_async(texts, tags_list)

    assert len(point_ids) == 5
    assert all(isinstance(pid, str) for pid in point_ids)

    results = await knowledge_tree.search_async("Knowledge point", k=10)
    assert len(results) >= 5

    for point_id in point_ids:
        await knowledge_tree.delete_entry_async(point_id)

    results = await knowledge_tree.search_async("Knowledge point", k=10)
    assert not any(r["id"] in point_ids for r in results)


@pytest.mark.asyncio
async def test_experience_tree_batch_operations(experience_tree):
    texts = [f"Experience {i}" for i in range(5)]
    tags_list = [[f"index:{i}"] for i in range(5)]

    point_ids = await experience_tree.add_entries_batch_async(texts, tags_list)

    assert len(point_ids) == 5
    assert all(isinstance(pid, str) for pid in point_ids)

    results = await experience_tree.search_async("Experience", k=10)
    assert len(results) >= 5

    for point_id in point_ids:
        await experience_tree.delete_entry_async(point_id)

    results = await experience_tree.search_async("Experience", k=10)
    assert not any(r["id"] in point_ids for r in results)


@pytest.mark.asyncio
async def test_knowledge_tree_update_point(knowledge_tree):
    point_id = await knowledge_tree.add_entry_async(
        text="Original content",
        tags=["version:1"],
    )

    point = await knowledge_tree.get_entry_async(point_id)
    assert point is not None

    new_point_id = await knowledge_tree.add_entry_async(
        text="Updated content",
        tags=["version:2"],
    )

    await knowledge_tree.add_relation_async(point_id, new_point_id, "updated_to")

    results = await knowledge_tree.search_async("Updated content")
    assert len(results) > 0

    updated_point = next((r for r in results if r["id"] == new_point_id), None)
    assert updated_point is not None
    assert "version:2" in updated_point.get("tags", [])


@pytest.mark.asyncio
async def test_experience_tree_update_point(experience_tree):
    point_id = await experience_tree.add_entry_async(
        text="Original experience",
        tags=["version:1"],
    )

    point = await experience_tree.get_entry_async(point_id)
    assert point is not None

    new_point_id = await experience_tree.add_entry_async(
        text="Updated experience",
        tags=["version:2"],
    )

    await experience_tree.add_relation_async(point_id, new_point_id, "updated_to")

    results = await experience_tree.search_async("Updated experience")
    assert len(results) > 0

    updated_point = next((r for r in results if r["id"] == new_point_id), None)
    assert updated_point is not None
    assert "version:2" in updated_point.get("tags", [])


@pytest.mark.asyncio
async def test_knowledge_tree_get_point(knowledge_tree):
    original_content = "Specific knowledge content"
    point_id = await knowledge_tree.add_entry_async(
        text=original_content,
        tags=["specific:true"],
    )

    point = await knowledge_tree.get_entry_async(point_id)

    assert point is not None
    assert point["id"] == point_id
    assert "specific:true" in point.get("tags", [])


@pytest.mark.asyncio
async def test_experience_tree_get_point(experience_tree):
    original_content = "Specific experience content"
    point_id = await experience_tree.add_entry_async(
        text=original_content,
        tags=["specific:true"],
    )

    point = await experience_tree.get_entry_async(point_id)

    assert point is not None
    assert point["id"] == point_id
    assert "specific:true" in point.get("tags", [])


@pytest.mark.asyncio
async def test_knowledge_tree_search_with_filters(knowledge_tree):
    await knowledge_tree.add_entry_async(
        text="Content about science",
        tags=["category:science"],
    )

    await knowledge_tree.add_entry_async(
        text="Content about history",
        tags=["category:history"],
    )

    await knowledge_tree.add_entry_async(
        text="More science content",
        tags=["category:science"],
    )

    science_results = await knowledge_tree.search_async("science")
    assert len(science_results) >= 2

    history_results = await knowledge_tree.search_async("history")
    assert len(history_results) >= 1


@pytest.mark.asyncio
async def test_experience_tree_search_with_filters(experience_tree):
    await experience_tree.add_entry_async(
        text="Experience with success",
        tags=["outcome:success"],
    )

    await experience_tree.add_entry_async(
        text="Experience with failure",
        tags=["outcome:failure"],
    )

    await experience_tree.add_entry_async(
        text="Another successful experience",
        tags=["outcome:success"],
    )

    success_results = await experience_tree.search_async("success")
    assert len(success_results) >= 2

    failure_results = await experience_tree.search_async("failure")
    assert len(failure_results) >= 1


@pytest.mark.asyncio
async def test_knowledge_tree_clear_all_points(knowledge_tree):
    for i in range(3):
        await knowledge_tree.add_entry_async(
            text=f"Test content {i}",
            tags=[f"index:{i}"],
        )

    await knowledge_tree.clear_all_points_async()

    results = await knowledge_tree.search_async("Test content")
    assert len(results) == 0
