from unittest import TestCase
from unittest.mock import patch

from langchain.messages import AIMessage, ToolMessage

from agent import ask, stream_ask


class AgentStreamTest(TestCase):
    def test_sql_progress_arrives_before_the_answer(self):
        query = "SELECT COUNT(*) FROM player_games"
        updates = [
            AIMessage(content="", tool_calls=[
                {"name": "sql_db_query", "args": {"query": query}, "id": "count"}
            ]),
            ToolMessage(content="[(25,)]", name="sql_db_query", tool_call_id="count"),
            AIMessage(content="There are 25 player-game rows."),
        ]

        def snapshots(inputs, **kwargs):
            messages = inputs["messages"].copy()
            yield {"messages": messages}
            for update in updates:
                messages = [*messages, update]
                yield {"messages": messages}

        question = "How many player-game rows are there?"
        with patch("agent._get_agent") as get_agent:
            get_agent.return_value.stream.side_effect = snapshots
            stream = stream_ask(question, [])
            self.assertEqual(next(stream), ("", [], "Reading your question."))
            get_agent.assert_not_called()
            answer, queries, activity = next(stream)
            self.assertEqual((answer, queries), ("", []))
            self.assertIn(f"Running read-only SQL\n{query}", activity)
            answer, queries, activity = next(stream)
            self.assertEqual((answer, queries), ("", [query]))
            self.assertIn("Database result\n[(25,)]", activity)
            answer, queries, _ = next(stream)
            self.assertEqual(answer, updates[-1].text)
            self.assertEqual(ask(question, []), (answer, queries))
