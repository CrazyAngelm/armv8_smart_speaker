from enum import Enum
from models import Event


class BaseEvent:

    def __init__(
        self, 
        session,
        id=None,
        *args, **kwargs,
    ):

        self.intent = self.__class__.__name__

        if not id:
            try:
                new_event = Event(**self.__dict__)
                session.add(new_event)
                session.commit()

                self.model = new_event

            except Exception as ex:
                print(ex)
                return None
        
        else:
            self.model = session.get(Event, id)


class SetTimerEvent(BaseEvent):

    def __init__(
        self, 
        timestamp=None,
        *args, **kwargs,
    ):

        self.status = 0
        self.timestamp = timestamp

        super().__init__(*args, **kwargs)


    def finish_event(
        self, 
        session,
    ) -> str:

        self.model.status = 1
        session.commit()

        return 'Таймер активирован'


class SetNotificationEvent(BaseEvent):

    def __init__(
        self, 
        timestamp=None, 
        notification_text=None,
        *args, **kwargs,
    ):

        self.status = 0
        self.timestamp = timestamp
        self.notification_text = notification_text

        super().__init__(*args, **kwargs)


    def finish_event(
        self, 
        session,
    ) -> str:

        self.model.status = 1
        session.commit()

        return 'Ваше напоминание: ' + self.notification_text
