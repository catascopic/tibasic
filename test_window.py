"""Window/graphing variable side effects and the Zoom memory."""
import pytest

from environment import Environment, Window, TableVars
from catalog import get_token
from errors import DomainError
from test_tibasic import run

# Token codes for the window/table variables (from catalog).
DELTA_X, DELTA_Y = 0x6326, 0x6327
XRES             = 0x6336
NMIN, NMAX       = 0x631F, 0x631D
XFACT, YFACT     = 0x6328, 0x6329
TBL_START        = 0x631A


def store(env, code, value):
	"""Store through the exact path the parser uses for `value→var`."""
	get_token(code).variable(env).store(value)


def resolve(env, code):
	return get_token(code).variable(env).resolve()


class TestDeltaDerived:
	def test_default_is_bounds_over_94(self):
		env = Environment()
		assert resolve(env, DELTA_X) == pytest.approx((10 - -10) / 94)

	def test_delta_y_over_62(self):
		env = Environment()
		assert resolve(env, DELTA_Y) == pytest.approx((10 - -10) / 62)

	def test_tracks_live_bounds(self):
		env = Environment()
		env.window.xmin.store(0.0)
		env.window.xmax.store(94.0)
		assert resolve(env, DELTA_X) == pytest.approx(1.0)

	def test_store_delta_x_moves_xmax_only(self):
		env = Environment()
		store(env, DELTA_X, 1.0)                      # xmax := xmin + 94·ΔX
		assert env.window.xmax.resolve() == pytest.approx(-10 + 94)
		assert env.window.xmin.resolve() == -10        # lower bound untouched

	def test_store_delta_y_moves_ymax_only(self):
		env = Environment()
		store(env, DELTA_Y, 2.0)
		assert env.window.ymax.resolve() == pytest.approx(-10 + 62 * 2)
		assert env.window.ymin.resolve() == -10

	def test_round_trips(self):
		env = Environment()
		store(env, DELTA_X, 0.5)
		assert resolve(env, DELTA_X) == pytest.approx(0.5)


class TestXres:
	def test_accepts_integer_in_range(self):
		env = Environment()
		store(env, XRES, 4.0)
		assert resolve(env, XRES) == 4.0

	@pytest.mark.parametrize("bad", [0.0, 9.0, -1.0, 1.5])
	def test_rejects_out_of_range_or_non_integer(self, bad):
		env = Environment()
		with pytest.raises(DomainError):
			store(env, XRES, bad)


class TestNMinNMax:
	def test_accepts_integers(self):
		env = Environment()
		store(env, NMIN, 3.0)
		store(env, NMAX, 20.0)
		assert (resolve(env, NMIN), resolve(env, NMAX)) == (3.0, 20.0)

	def test_rejects_non_integer(self):
		env = Environment()
		with pytest.raises(DomainError):
			store(env, NMAX, 2.5)

	def test_stays_float(self):
		env = Environment()
		store(env, NMIN, 5.0)
		assert type(resolve(env, NMIN)) is float


class TestZoomFactors:
	@pytest.mark.parametrize("code", [XFACT, YFACT])
	def test_accepts_at_least_one(self, code):
		env = Environment()
		store(env, code, 1.0)        # boundary
		store(env, code, 7.5)
		assert resolve(env, code) == 7.5

	@pytest.mark.parametrize("code", [XFACT, YFACT])
	def test_rejects_below_one(self, code):
		env = Environment()
		with pytest.raises(DomainError):
			store(env, code, 0.5)


class TestZoomMemory:
	def test_sto_then_rcl_restores_window(self):
		env = Environment()
		env.window.xmax.store(50.0)
		run('ZoomSto', env)
		env.window.xmax.store(99.0)
		run('ZoomRcl', env)
		assert env.window.xmax.resolve() == 50.0

	def test_rcl_restores_derived_delta(self):
		env = Environment()
		env.window.xmin.store(0.0)
		env.window.xmax.store(94.0)   # ΔX = 1
		run('ZoomSto', env)
		env.window.xmax.store(10.0)
		run('ZoomRcl', env)
		assert resolve(env, DELTA_X) == pytest.approx(1.0)

	def test_store_snapshot_is_independent(self):
		env = Environment()
		env.window.xmax.store(50.0)
		run('ZoomSto', env)
		env.window.xmax.store(99.0)
		assert env.zoom_window.xmax.resolve() == 50.0   # snapshot unchanged

	def test_copy_excludes_derived_but_recomputes(self):
		w = Window()
		w.xmin.store(0.0)
		w.xmax.store(62.0)
		clone = w.copy()
		assert clone.delta_x.resolve() == pytest.approx(62 / 94)
		assert clone.delta_x.window is clone        # bound to the clone, not the original


class TestTableVarsSeparated:
	def test_table_token_routes_to_table_not_window(self):
		env = Environment()
		store(env, TBL_START, 5.0)
		assert env.table.tbl_start.resolve() == 5.0
		assert not hasattr(env.window, 'tbl_start')

	def test_tbl_input_is_a_list_slot(self):
		from core import ListVariable
		assert isinstance(TableVars().tbl_input, ListVariable)


class TestWiring:
	def test_zoom_commands_are_wired(self):
		assert get_token(0x92).command is not None   # ZoomSto
		assert get_token(0x90).command is not None   # ZoomRcl
